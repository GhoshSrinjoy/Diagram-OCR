"""Download the non-HF-API datasets listed in sources_v2.yaml (git repos, Zenodo, plain URLs).

Why a second downloader: download.py fetches Hugging Face repos file by file through the API,
which is ideal for a few large files but hits the 5,000 requests / 5 min limit on repos with
tens of thousands of small files (FlowGen has 60,481). Those, plus GitHub / Zenodo / GCS
sources, are fetched here instead.

    method: git     git clone --depth 1 (LFS pulled in batches), then optionally drop .git
    method: zenodo  resolve files via the Zenodo REST API, resumable download, md5-verified
    method: url     resumable download of each URL

Every finished dataset gets <local_dir>/.download_complete.json and an entry in
<output_dir>/sources-lock.json; finished datasets are skipped on re-runs.

Usage:
    python download_sources.py --list
    python download_sources.py                         # dry run: plan + sizes
    python download_sources.py --download              # everything not yet complete
    python download_sources.py --download --dataset didi --dataset cghd
    python download_sources.py --download --replace-partial   # wipe half-finished folders first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
import yaml

DEFAULT_MANIFEST = Path(__file__).with_name("sources_v2.yaml")
MARKER = ".download_complete.json"
CHUNK = 4 * 1024 * 1024


@dataclass
class Source:
    key: str
    name: str
    method: str
    local_dir: str
    license: str
    use: str
    expected_size_bytes: Optional[int] = None
    url: Optional[str] = None
    urls: list[str] = field(default_factory=list)
    record: Optional[int] = None
    files: list[str] = field(default_factory=list)
    strip_git: bool = False
    notes: str = ""


def load_manifest(path: Path) -> tuple[Path, list[Source]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out_dir = Path(raw["storage"]["output_dir"])
    sources = []
    for e in raw["datasets"]:
        if e["method"] not in ("git", "zenodo", "url"):
            raise SystemExit(f"{e['key']}: unknown method {e['method']!r}")
        sources.append(Source(**{k: v for k, v in e.items() if k in Source.__dataclass_fields__}))
    return out_dir, sources


def human(n: Optional[float]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def dir_stats(path: Path) -> tuple[int, int]:
    total = count = 0
    for p in path.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            total += p.stat().st_size
            count += 1
    return total, count


def _rmtree(path: Path) -> None:
    def onerror(func, p, _exc):  # git marks pack files read-only on Windows
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onerror=onerror)


# --------------------------------------------------------------------------- #
# Fetchers
# --------------------------------------------------------------------------- #
def fetch_url(session: requests.Session, url: str, dest: Path, size: Optional[int] = None,
              md5: Optional[str] = None) -> None:
    """Resumable streaming download to dest (via dest.part)."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        print(f"    have {dest.name}")
        return
    part = dest.with_name(dest.name + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with session.get(url, stream=True, headers=headers, timeout=60) as r:
        if pos and r.status_code != 206:  # server ignored Range: start over
            pos = 0
        r.raise_for_status()
        total = size or (pos + int(r.headers.get("Content-Length", 0)) or None)
        t0, done = time.time(), pos
        with open(part, "ab" if pos else "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                rate = (done - pos) / max(time.time() - t0, 1e-6)
                sys.stdout.write(f"\r    {dest.name}: {human(done)} / {human(total)}  {human(rate)}/s   ")
                sys.stdout.flush()
    print()
    if size is not None and part.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: got {part.stat().st_size} bytes, expected {size}")
    if md5:
        h = hashlib.md5()
        with open(part, "rb") as fh:
            for block in iter(lambda: fh.read(CHUNK), b""):
                h.update(block)
        if h.hexdigest() != md5:
            raise RuntimeError(f"{dest.name}: md5 mismatch")
    part.replace(dest)


def fetch_git(src: Source, dest: Path) -> None:
    # Clone pointers only, then fetch LFS objects in one batched, parallel pull; letting the
    # checkout smudge files one at a time runs at ~2 MB/s.
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    cmd = ["git", "clone", "-q", "--depth", "1", src.url, str(dest)]
    print("    " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    subprocess.run(["git", "-C", str(dest), "-c", "lfs.concurrenttransfers=16", "lfs", "pull"],
                   check=True, env=env)
    if src.strip_git:
        _rmtree(dest / ".git")  # LFS keeps a second copy of every file in .git/lfs


def zenodo_files(session: requests.Session, src: Source) -> list[dict[str, Any]]:
    j = session.get(f"https://zenodo.org/api/records/{src.record}", timeout=60).json()
    by_key = {f["key"]: f for f in j["files"]}
    missing = [n for n in src.files if n not in by_key]
    if missing:
        raise RuntimeError(f"{src.key}: files not in Zenodo record {src.record}: {missing}")
    return [by_key[n] for n in src.files]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def is_complete(dest: Path) -> bool:
    return (dest / MARKER).exists()


def write_lock(out_dir: Path, src: Source, dest: Path) -> None:
    total, count = dir_stats(dest)
    info = {
        "key": src.key, "name": src.name, "method": src.method,
        "source": src.url or src.urls or f"zenodo:{src.record}",
        "license": src.license, "use": src.use, "bytes": total, "files": count,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (dest / MARKER).write_text(json.dumps(info, indent=2), encoding="utf-8")
    lock_path = out_dir / "sources-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    lock[src.key] = info
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"    done: {count} files, {human(total)}")


def download(src: Source, out_dir: Path, session: requests.Session, replace_partial: bool) -> None:
    dest = out_dir / src.local_dir
    if is_complete(dest):
        print(f"[{src.key}] already complete")
        return
    print(f"[{src.key}] {src.name}  ({human(src.expected_size_bytes)})")
    if src.method == "git":
        if dest.exists() and any(dest.iterdir()):
            if not replace_partial:
                raise SystemExit(f"{dest} exists but is incomplete; re-run with --replace-partial")
            print(f"    removing partial {dest}")
            _rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fetch_git(src, dest)
    elif src.method == "zenodo":
        dest.mkdir(parents=True, exist_ok=True)
        for f in zenodo_files(session, src):
            md5 = f["checksum"].split(":", 1)[1] if f["checksum"].startswith("md5:") else None
            fetch_url(session, f["links"]["self"], dest / f["key"], f["size"], md5)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        for url in src.urls:
            fetch_url(session, url, dest / url.rsplit("/", 1)[1])
    write_lock(out_dir, src, dest)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dataset", action="append", default=[], help="key to download (repeatable)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", action="store_true", help="actually download (default: dry run)")
    ap.add_argument("--replace-partial", action="store_true", help="delete half-finished git folders")
    args = ap.parse_args(argv)

    out_dir, sources = load_manifest(args.manifest)
    if args.dataset:
        unknown = set(args.dataset) - {s.key for s in sources}
        if unknown:
            raise SystemExit(f"unknown dataset keys: {sorted(unknown)}")
        sources = [s for s in sources if s.key in args.dataset]

    if args.list or not args.download:
        total = 0
        for s in sources:
            state = "complete" if is_complete(out_dir / s.local_dir) else "-"
            total += s.expected_size_bytes or 0
            size = human(s.expected_size_bytes)
            print(f"{s.key:22} {s.method:7} {size:>9}  {s.use:9} {state:9} {s.license}")
        print(f"{'total':22} {'':7} {human(total):>9}  -> {out_dir}")
        if not args.download and not args.list:
            print("Dry run. Add --download to fetch.")
        return 0

    free = shutil.disk_usage(out_dir.anchor).free
    need = sum(s.expected_size_bytes or 0 for s in sources if not is_complete(out_dir / s.local_dir))
    if need * 2 + 5 * 1024**3 > free:  # x2: git clones hold a second copy until .git is removed
        raise SystemExit(f"not enough disk: need ~{human(need * 2)} + 5 GB headroom, free {human(free)}")

    session = requests.Session()
    failed = []
    for s in sources:
        try:
            download(s, out_dir, session, args.replace_partial)
        except (subprocess.CalledProcessError, requests.RequestException, RuntimeError) as exc:
            print(f"    FAILED: {exc}")
            failed.append(s.key)
    if failed:
        print(f"failed: {failed} (re-run to resume)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
