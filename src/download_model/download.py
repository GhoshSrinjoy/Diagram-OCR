from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_ROOT = Path("D:/data")
DEFAULT_MANIFEST = Path(__file__).with_name("model.yaml")


@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    repo_id: str
    repo_type: str
    revision: str
    stage: str
    priority: str
    family: str
    task: str
    enabled: bool
    gated: bool
    recommended: bool
    license: str
    allow_patterns: tuple[str, ...]
    ignore_patterns: tuple[str, ...]
    expected_size_bytes: int | None
    max_allowed_bytes: int | None
    local_dir: str
    training_role: str
    notes: str


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int | None
    blob_id: str | None = None
    lfs_oid: str | None = None


@dataclass(frozen=True)
class ModelPlan:
    config: ModelConfig
    resolved_revision: str
    target_dir: Path
    files: tuple[RemoteFile, ...]

    @property
    def total_size(self) -> int | None:
        total = 0
        for file in self.files:
            if file.size is None:
                return None
            total += file.size
        return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manifest-driven Hugging Face model downloader. By default it checks "
            "remote file sizes and prints a Rich table. Use --download to fetch files."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model key or repo_id from model.yaml. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--all", action="store_true", help="Select all enabled models.")
    parser.add_argument("--list", action="store_true", help="List model manifest entries and exit.")
    parser.add_argument("--check-size", action="store_true", help="Resolve remote sizes and print the plan.")
    parser.add_argument("--download", action="store_true", help="Actually download selected models.")
    parser.add_argument("--verify-only", action="store_true", help="Check local files and sizes only.")
    parser.add_argument("--manifest-only", action="store_true", help="Use YAML expected sizes only; no network.")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--allow-outside-data-dir", action="store_true")
    parser.add_argument("--ignore-size-limits", action="store_true")
    parser.add_argument("--allow-unknown-size", action="store_true")
    parser.add_argument("--skip-disk-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace complete files with wrong sizes.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing .part files.")
    parser.add_argument("--token", default=None, help="Hugging Face token. Defaults to HF_TOKEN.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file to load before reading HF_TOKEN. Defaults include src/.env.",
    )
    parser.add_argument("--revision", default=None, help="Override revision for every selected model.")
    parser.add_argument("--max-total-gb", type=float, default=None)
    parser.add_argument("--min-free-gb", type=float, default=None)
    parser.add_argument("--chunk-size-mb", type=float, default=8.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def load_runtime_deps(require_hf: bool, require_requests: bool) -> dict[str, Any]:
    missing: list[str] = []

    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore
        missing.append("pyyaml")

    try:
        from rich.console import Console  # type: ignore
        from rich.filesize import decimal as rich_format_size  # type: ignore
        from rich.progress import (  # type: ignore
            BarColumn,
            DownloadColumn,
            Progress,
            ProgressColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )
        from rich.table import Table  # type: ignore
        from rich.text import Text  # type: ignore
    except ImportError:
        Console = None  # type: ignore
        rich_format_size = None  # type: ignore
        BarColumn = None  # type: ignore
        DownloadColumn = None  # type: ignore
        Progress = None  # type: ignore
        ProgressColumn = None  # type: ignore
        TaskProgressColumn = None  # type: ignore
        TextColumn = None  # type: ignore
        TimeElapsedColumn = None  # type: ignore
        TimeRemainingColumn = None  # type: ignore
        TransferSpeedColumn = None  # type: ignore
        Table = None  # type: ignore
        Text = None  # type: ignore
        missing.append("rich")

    HfApi = None
    hf_hub_url = None
    HfHubHTTPError = Exception
    if require_hf:
        try:
            from huggingface_hub import HfApi, hf_hub_url  # type: ignore
            try:
                from huggingface_hub.errors import HfHubHTTPError  # type: ignore
            except ImportError:
                from huggingface_hub.utils import HfHubHTTPError  # type: ignore
        except ImportError:
            missing.append("huggingface_hub")

    requests = None
    HTTPAdapter = None
    Retry = None
    if require_requests:
        try:
            import requests  # type: ignore
            from requests.adapters import HTTPAdapter  # type: ignore
            from urllib3.util.retry import Retry  # type: ignore
        except ImportError:
            missing.append("requests")

    if missing:
        packages = " ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing required packages: "
            f"{packages}\nInstall them with:\n"
            f"  python -m pip install {packages}"
        )

    return {
        "yaml": yaml,
        "Console": Console,
        "format_size": rich_format_size,
        "BarColumn": BarColumn,
        "DownloadColumn": DownloadColumn,
        "Progress": Progress,
        "ProgressColumn": ProgressColumn,
        "TaskProgressColumn": TaskProgressColumn,
        "TextColumn": TextColumn,
        "Text": Text,
        "TimeElapsedColumn": TimeElapsedColumn,
        "TimeRemainingColumn": TimeRemainingColumn,
        "TransferSpeedColumn": TransferSpeedColumn,
        "Table": Table,
        "HfApi": HfApi,
        "hf_hub_url": hf_hub_url,
        "HfHubHTTPError": HfHubHTTPError,
        "requests": requests,
        "HTTPAdapter": HTTPAdapter,
        "Retry": Retry,
    }


def parse_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        raise ValueError(f"Unsupported size value: {value!r}")

    text = value.strip()
    match = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?", text)
    if not match:
        raise ValueError(f"Could not parse size value: {value!r}")

    amount = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    multipliers = {
        "B": 1,
        "K": 1_000,
        "KB": 1_000,
        "M": 1_000_000,
        "MB": 1_000_000,
        "G": 1_000_000_000,
        "GB": 1_000_000_000,
        "T": 1_000_000_000_000,
        "TB": 1_000_000_000_000,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported size unit in {value!r}")
    return int(amount * multipliers[unit])


def bytes_from_gb(value: float | None) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError("GB values must be positive")
    return int(value * 1_000_000_000)


def format_size(deps: dict[str, Any], value: int | None) -> str:
    if value is None:
        return "unknown"
    return deps["format_size"](value)


def candidate_env_files(explicit_env_file: Path | None) -> list[Path]:
    if explicit_env_file is not None:
        return [explicit_env_file]

    script_path = Path(__file__).resolve()
    candidates = [
        Path.cwd() / ".env",
        script_path.parent / ".env",
        script_path.parents[1] / ".env",
        script_path.parents[2] / ".env",
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    else:
        value = value.split(" #", 1)[0].strip()
    return key, value


def load_env_file(path: Path, override: bool = False) -> int:
    loaded = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            parsed = parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if not override and key in os.environ:
                continue
            os.environ[key] = value
            loaded += 1
    return loaded


def load_dotenv(args: argparse.Namespace, deps: dict[str, Any]) -> Path | None:
    for path in candidate_env_files(args.env_file):
        if not path.exists():
            continue
        load_env_file(path)
        if args.env_file is not None or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            deps["Console"]().print(f"[dim]Loaded environment from {path}[/dim]")
            return path
    if args.env_file is not None:
        raise SystemExit(f"Requested --env-file does not exist: {args.env_file}")
    return None


def resolve_hf_token(args: argparse.Namespace) -> str | None:
    return args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def read_manifest(path: Path, yaml_module: Any) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml_module.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Manifest must be a YAML mapping: {path}")
    return data


def unique_patterns(patterns: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            result.append(pattern)
    return tuple(result)


def build_model_configs(raw: dict[str, Any]) -> list[ModelConfig]:
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SystemExit("Manifest defaults must be a mapping")

    entries = raw.get("models") or []
    if not isinstance(entries, list):
        raise SystemExit("Manifest models must be a list")

    configs: list[ModelConfig] = []
    keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit("Every model entry must be a mapping")

        key = str(entry.get("key", "")).strip()
        repo_id = str(entry.get("repo_id", "")).strip()
        if not key or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
            raise SystemExit(f"Invalid model key: {key!r}")
        if key in keys:
            raise SystemExit(f"Duplicate model key: {key}")
        keys.add(key)
        if not repo_id or "/" not in repo_id:
            raise SystemExit(f"Model {key} has invalid repo_id: {repo_id!r}")

        default_ignores = list(defaults.get("ignore_patterns") or [])
        entry_ignores = list(entry.get("ignore_patterns") or [])
        allow_patterns = entry.get("allow_patterns") or defaults.get("allow_patterns") or ["**"]
        expected_size = entry.get("expected_size_bytes", entry.get("expected_size"))
        max_allowed_size = entry.get("max_allowed_size_bytes", entry.get("max_allowed_size"))

        configs.append(
            ModelConfig(
                key=key,
                name=str(entry.get("name") or key),
                repo_id=repo_id,
                repo_type=str(entry.get("repo_type") or defaults.get("repo_type") or "model"),
                revision=str(entry.get("revision") or defaults.get("revision") or "main"),
                stage=str(entry.get("stage") or defaults.get("stage") or "smoke_baseline"),
                priority=str(entry.get("priority") or defaults.get("priority") or "high"),
                family=str(entry.get("family") or "unknown"),
                task=str(entry.get("task") or "unknown"),
                enabled=bool(entry.get("enabled", defaults.get("enabled", True))),
                gated=bool(entry.get("gated", defaults.get("gated", False))),
                recommended=bool(entry.get("recommended", False)),
                license=str(entry.get("license") or "unknown"),
                allow_patterns=unique_patterns([str(pattern) for pattern in allow_patterns]),
                ignore_patterns=unique_patterns([str(pattern) for pattern in default_ignores + entry_ignores]),
                expected_size_bytes=parse_size(expected_size),
                max_allowed_bytes=parse_size(max_allowed_size),
                local_dir=str(entry.get("local_dir") or repo_id.replace("/", "__")),
                training_role=str(entry.get("training_role") or ""),
                notes=str(entry.get("notes") or ""),
            )
        )
    return configs


def split_selection_tokens(values: list[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        for token in value.split(","):
            cleaned = token.strip()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def active_configs(configs: list[ModelConfig], args: argparse.Namespace) -> list[ModelConfig]:
    result: list[ModelConfig] = []
    for config in configs:
        if args.stage and config.stage != args.stage:
            continue
        if not config.enabled and not args.include_disabled:
            continue
        if config.gated and not args.include_gated:
            continue
        result.append(config)
    return result


def select_models(configs: list[ModelConfig], args: argparse.Namespace) -> list[ModelConfig]:
    active = active_configs(configs, args)
    if not active:
        raise SystemExit("No models available after filters.")

    tokens = split_selection_tokens(args.model)
    if args.all:
        tokens = ["all"]
    if not tokens:
        recommended = [config for config in active if config.recommended]
        return recommended or active[:1]

    if any(token.lower() in {"a", "all", "*"} for token in tokens):
        return active

    by_key = {config.key.lower(): config for config in active}
    by_repo = {config.repo_id.lower(): config for config in active}
    selected: list[ModelConfig] = []
    selected_keys: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        config = by_key.get(lowered) or by_repo.get(lowered)
        if config is None:
            available = ", ".join(config.key for config in active)
            raise SystemExit(f"Unknown model selection {token!r}. Available keys: {available}")
        if config.key not in selected_keys:
            selected_keys.add(config.key)
            selected.append(config)
    return selected


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_output_dir(raw_manifest: dict[str, Any], args: argparse.Namespace) -> Path:
    storage = raw_manifest.get("storage") or {}
    manifest_output = storage.get("output_dir") if isinstance(storage, dict) else None
    output_dir = args.output_dir or Path(str(manifest_output or DATA_ROOT / "diagram_extraction" / "models" / "base"))
    output_dir = output_dir.expanduser().resolve()
    data_root = DATA_ROOT.resolve()
    if not args.allow_outside_data_dir and not is_relative_to(output_dir, data_root):
        raise SystemExit(
            f"Refusing to write outside {data_root}. "
            "Use --allow-outside-data-dir if this is intentional."
        )
    return output_dir


def safe_target_dir(output_dir: Path, local_dir: str) -> Path:
    target = (output_dir / local_dir).resolve()
    if not is_relative_to(target, output_dir):
        raise SystemExit(f"Unsafe local_dir escapes output directory: {local_dir!r}")
    return target


def matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    normalized = path.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def selected_remote_files(config: ModelConfig, siblings: list[Any]) -> tuple[RemoteFile, ...]:
    files: list[RemoteFile] = []
    for sibling in siblings:
        path = getattr(sibling, "rfilename", None)
        if not path:
            continue
        path = str(path).replace("\\", "/")
        if not matches_any(path, config.allow_patterns):
            continue
        if matches_any(path, config.ignore_patterns):
            continue

        size = getattr(sibling, "size", None)
        blob_id = getattr(sibling, "blob_id", None)
        lfs_oid = None
        lfs = getattr(sibling, "lfs", None)
        if isinstance(lfs, dict):
            size = size if size is not None else lfs.get("size")
            lfs_oid = lfs.get("oid")
        elif lfs is not None:
            size = size if size is not None else getattr(lfs, "size", None)
            lfs_oid = getattr(lfs, "oid", None)

        files.append(RemoteFile(path=path, size=int(size) if size is not None else None, blob_id=blob_id, lfs_oid=lfs_oid))
    return tuple(sorted(files, key=lambda item: item.path))


def resolve_plans(
    deps: dict[str, Any],
    selected: list[ModelConfig],
    output_dir: Path,
    token: str | None,
    revision_override: str | None,
) -> list[ModelPlan]:
    api = deps["HfApi"](token=token)
    plans: list[ModelPlan] = []
    console = deps["Console"]()

    for config in selected:
        revision = revision_override or config.revision
        console.print(f"[bold]Resolving[/bold] {config.repo_id}@{revision}")
        try:
            repo_info = api.repo_info(
                repo_id=config.repo_id,
                repo_type=config.repo_type,
                revision=revision,
                files_metadata=True,
            )
        except deps["HfHubHTTPError"] as exc:
            raise SystemExit(f"Could not resolve {config.repo_id}@{revision}: {exc}") from exc

        files = selected_remote_files(config, list(getattr(repo_info, "siblings", [])))
        if not files:
            raise SystemExit(f"No files selected for {config.repo_id}. Check allow/ignore patterns.")
        plans.append(
            ModelPlan(
                config=config,
                resolved_revision=str(getattr(repo_info, "sha", revision)),
                target_dir=safe_target_dir(output_dir, config.local_dir),
                files=files,
            )
        )
    return plans


def total_plan_size(plans: list[ModelPlan]) -> int | None:
    total = 0
    for plan in plans:
        size = plan.total_size
        if size is None:
            return None
        total += size
    return total


def enforce_size_limits(
    deps: dict[str, Any],
    plans: list[ModelPlan],
    raw_manifest: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.ignore_size_limits:
        return

    unknown = [(plan, file) for plan in plans for file in plan.files if file.size is None]
    if unknown and not args.allow_unknown_size:
        sample = ", ".join(f"{plan.config.key}:{file.path}" for plan, file in unknown[:5])
        raise SystemExit(
            "Some remote file sizes are unknown, so model-size guardrails cannot run. "
            f"Examples: {sample}. Use --allow-unknown-size if this is intentional."
        )

    for plan in plans:
        total_size = plan.total_size
        max_allowed = plan.config.max_allowed_bytes
        if total_size is not None and max_allowed is not None and total_size > max_allowed:
            raise SystemExit(
                f"{plan.config.key} resolved to {format_size(deps, total_size)}, "
                f"above its manifest limit of {format_size(deps, max_allowed)}. "
                "Update model.yaml or use --ignore-size-limits after reviewing the files."
            )

    storage = raw_manifest.get("storage") or {}
    manifest_limit = parse_size(storage.get("max_total_size")) if isinstance(storage, dict) else None
    cli_limit = bytes_from_gb(args.max_total_gb)
    max_total = cli_limit if cli_limit is not None else manifest_limit
    total = total_plan_size(plans)
    if max_total is not None and total is not None and total > max_total:
        raise SystemExit(
            f"Selected models total {format_size(deps, total)}, above the total limit "
            f"of {format_size(deps, max_total)}."
        )


def local_progress_bytes(plan: ModelPlan, file: RemoteFile, overwrite: bool, resume: bool) -> int:
    target = plan.target_dir / file.path
    part = target.with_name(target.name + ".part")
    if file.size is None:
        return 0
    if target.exists() and target.stat().st_size == file.size:
        return file.size
    if not overwrite and resume and part.exists():
        return min(part.stat().st_size, file.size)
    return 0


def bytes_needed(plans: list[ModelPlan], overwrite: bool, resume: bool) -> int | None:
    needed = 0
    for plan in plans:
        for file in plan.files:
            if file.size is None:
                return None
            needed += max(file.size - local_progress_bytes(plan, file, overwrite, resume), 0)
    return needed


def check_disk_space(
    deps: dict[str, Any],
    plans: list[ModelPlan],
    output_dir: Path,
    raw_manifest: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.skip_disk_check:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    needed = bytes_needed(plans, overwrite=args.overwrite, resume=not args.no_resume)
    if needed is None:
        if args.allow_unknown_size:
            return
        raise SystemExit("Cannot check disk space because at least one selected file has unknown size.")

    storage = raw_manifest.get("storage") or {}
    manifest_min_free = parse_size(storage.get("min_free_after_download")) if isinstance(storage, dict) else None
    cli_min_free = bytes_from_gb(args.min_free_gb)
    min_free = cli_min_free if cli_min_free is not None else (manifest_min_free or 0)
    usage = shutil.disk_usage(output_dir)
    free_after = usage.free - needed
    if free_after < min_free:
        raise SystemExit(
            f"Not enough disk space for {output_dir}. Need {format_size(deps, needed)}, "
            f"free now is {format_size(deps, usage.free)}, and guardrail requires "
            f"{format_size(deps, min_free)} free after download."
        )


def diff_text(deps: dict[str, Any], expected: int | None, actual: int | None) -> str:
    if expected is None or actual is None:
        return "unknown"
    diff = actual - expected
    if diff == 0:
        return "0 B"
    sign = "+" if diff > 0 else "-"
    return f"{sign}{format_size(deps, abs(diff))}"


def status_text(actual: int | None, max_allowed: int | None) -> str:
    if actual is None or max_allowed is None:
        return "n/a"
    if actual > max_allowed:
        return "over limit"
    return "ok"


def print_manifest_list(deps: dict[str, Any], configs: list[ModelConfig]) -> None:
    console = deps["Console"]()
    table = deps["Table"](title="Configured Models")
    table.add_column("Key")
    table.add_column("Repo")
    table.add_column("Family")
    table.add_column("Expected", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Recommended")
    table.add_column("Enabled")
    for config in configs:
        table.add_row(
            config.key,
            config.repo_id,
            config.family,
            format_size(deps, config.expected_size_bytes),
            format_size(deps, config.max_allowed_bytes),
            "yes" if config.recommended else "no",
            "yes" if config.enabled else "no",
        )
    console.print(table)


def print_manifest_sizes(deps: dict[str, Any], selected: list[ModelConfig]) -> None:
    console = deps["Console"]()
    table = deps["Table"](title="Manifest Model Sizes")
    table.add_column("Key")
    table.add_column("Repo")
    table.add_column("Expected", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("License")
    table.add_column("Role")
    total = 0
    unknown = False
    for config in selected:
        if config.expected_size_bytes is None:
            unknown = True
        else:
            total += config.expected_size_bytes
        table.add_row(
            config.key,
            config.repo_id,
            format_size(deps, config.expected_size_bytes),
            format_size(deps, config.max_allowed_bytes),
            config.license,
            config.training_role,
        )
    table.caption = "Total expected: unknown" if unknown else f"Total expected: {format_size(deps, total)}"
    console.print(table)


def print_remote_plan(deps: dict[str, Any], plans: list[ModelPlan], output_dir: Path) -> None:
    console = deps["Console"]()
    table = deps["Table"](title=f"Resolved Model Sizes -> {output_dir}")
    table.add_column("Key")
    table.add_column("Repo")
    table.add_column("Revision")
    table.add_column("Files", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Remote Size", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Status")

    total_expected = 0
    total_actual = 0
    unknown_expected = False
    unknown_actual = False
    for plan in plans:
        expected = plan.config.expected_size_bytes
        actual = plan.total_size
        if expected is None:
            unknown_expected = True
        else:
            total_expected += expected
        if actual is None:
            unknown_actual = True
        else:
            total_actual += actual
        table.add_row(
            plan.config.key,
            plan.config.repo_id,
            plan.resolved_revision[:12],
            str(len(plan.files)),
            format_size(deps, expected),
            format_size(deps, actual),
            diff_text(deps, expected, actual),
            format_size(deps, plan.config.max_allowed_bytes),
            status_text(actual, plan.config.max_allowed_bytes),
        )

    expected_text = "unknown" if unknown_expected else format_size(deps, total_expected)
    actual_text = "unknown" if unknown_actual else format_size(deps, total_actual)
    table.caption = f"Total expected: {expected_text} | Total remote: {actual_text}"
    console.print(table)


def verify_local_files(deps: dict[str, Any], plans: list[ModelPlan]) -> int:
    console = deps["Console"]()
    table = deps["Table"](title="Local Model Verification")
    table.add_column("Model")
    table.add_column("Files", justify="right")
    table.add_column("OK", justify="right")
    table.add_column("Missing", justify="right")
    table.add_column("Wrong Size", justify="right")
    table.add_column("Target")

    failures = 0
    for plan in plans:
        ok = 0
        missing = 0
        wrong = 0
        for file in plan.files:
            target = plan.target_dir / file.path
            if not target.exists():
                missing += 1
                continue
            if file.size is not None and target.stat().st_size != file.size:
                wrong += 1
                continue
            ok += 1
        failures += missing + wrong
        table.add_row(plan.config.key, str(len(plan.files)), str(ok), str(missing), str(wrong), str(plan.target_dir))
    console.print(table)
    return failures


class RemainingDownloadColumnFactory:
    def __init__(self, deps: dict[str, Any]) -> None:
        self._ProgressColumn = deps["ProgressColumn"]
        self._Text = deps["Text"]
        self._format_size = deps["format_size"]

    def make(self) -> Any:
        ProgressColumn = self._ProgressColumn
        Text = self._Text
        rich_format_size = self._format_size

        class RemainingDownloadColumn(ProgressColumn):
            def render(self, task: Any) -> Any:
                if task.total is None:
                    return Text("remaining: unknown")
                remaining = max(int(task.total - task.completed), 0)
                return Text(f"remaining: {rich_format_size(remaining)}")

        return RemainingDownloadColumn()


def make_session(deps: dict[str, Any], token: str | None) -> Any:
    requests = deps["requests"]
    Retry = deps["Retry"]
    HTTPAdapter = deps["HTTPAdapter"]

    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    session.headers.update({"User-Agent": "diagram-extraction-model-downloader/1.0"})
    return session


def remote_url(deps: dict[str, Any], plan: ModelPlan, file: RemoteFile) -> str:
    return deps["hf_hub_url"](
        repo_id=plan.config.repo_id,
        filename=file.path,
        repo_type=plan.config.repo_type,
        revision=plan.resolved_revision,
    )


def response_error(response: Any, plan: ModelPlan, file: RemoteFile) -> str:
    if response.status_code in {401, 403}:
        return (
            f"Access denied for {plan.config.repo_id}:{file.path}. "
            "Check model permissions and HF_TOKEN."
        )
    return f"HTTP {response.status_code} for {plan.config.repo_id}:{file.path}"


def download_file(
    deps: dict[str, Any],
    session: Any,
    plan: ModelPlan,
    file: RemoteFile,
    progress: Any,
    overall_task: int,
    current_task: int,
    args: argparse.Namespace,
) -> str:
    target = plan.target_dir / file.path
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if file.size is None or target.stat().st_size == file.size:
            completed = target.stat().st_size if file.size is None else file.size
            progress.advance(overall_task, completed)
            return "skipped"
        if not args.overwrite:
            raise RuntimeError(
                f"Existing file has wrong size: {target}. "
                f"Expected {file.size}, got {target.stat().st_size}. Use --overwrite to replace it."
            )
        target.unlink()

    if args.overwrite and part.exists():
        part.unlink()
    if args.no_resume and part.exists():
        part.unlink()

    resume_at = part.stat().st_size if part.exists() else 0
    headers: dict[str, str] = {}
    if resume_at > 0:
        headers["Range"] = f"bytes={resume_at}-"

    file_label = f"{plan.config.key}:{file.path}"
    if len(file_label) > 100:
        file_label = f"{plan.config.key}:...{file.path[-86:]}"
    progress.update(current_task, description=file_label, total=file.size, completed=0, visible=True)

    with session.get(remote_url(deps, plan, file), headers=headers, stream=True, timeout=(20, args.timeout_seconds)) as response:
        if resume_at > 0 and response.status_code == 416 and file.size is not None and resume_at == file.size:
            os.replace(part, target)
            progress.advance(overall_task, file.size)
            progress.update(current_task, completed=file.size)
            return "completed-from-part"

        if resume_at > 0 and response.status_code == 206:
            mode = "ab"
            progress.update(current_task, completed=resume_at)
            progress.advance(overall_task, resume_at)
        else:
            if response.status_code != 200:
                raise RuntimeError(response_error(response, plan, file))
            if part.exists():
                part.unlink()
            mode = "wb"

        response.raise_for_status()
        with part.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=int(args.chunk_size_mb * 1024 * 1024)):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.advance(current_task, len(chunk))
                progress.advance(overall_task, len(chunk))

    final_size = part.stat().st_size
    if file.size is not None and final_size != file.size:
        raise RuntimeError(f"Incomplete download for {target}: expected {file.size}, got {final_size}")
    os.replace(part, target)
    progress.update(current_task, completed=file.size if file.size is not None else final_size)
    return "downloaded"


def download_plans(
    deps: dict[str, Any],
    plans: list[ModelPlan],
    output_dir: Path,
    token: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(deps, token)
    total = total_plan_size(plans)
    remaining_column = RemainingDownloadColumnFactory(deps).make()

    results: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "models": [],
    }

    with deps["Progress"](
        deps["TextColumn"]("[bold blue]{task.description}"),
        deps["BarColumn"](),
        deps["TaskProgressColumn"](),
        deps["DownloadColumn"](binary_units=False),
        remaining_column,
        deps["TransferSpeedColumn"](),
        deps["TimeElapsedColumn"](),
        deps["TimeRemainingColumn"](),
    ) as progress:
        overall_task = progress.add_task("overall", total=total)
        current_task = progress.add_task("current", total=1, visible=False)

        for plan in plans:
            model_result = {
                "key": plan.config.key,
                "repo_id": plan.config.repo_id,
                "repo_type": plan.config.repo_type,
                "requested_revision": plan.config.revision,
                "resolved_revision": plan.resolved_revision,
                "target_dir": str(plan.target_dir),
                "files": [],
            }
            for file in plan.files:
                status = download_file(deps, session, plan, file, progress, overall_task, current_task, args)
                model_result["files"].append(
                    {
                        "path": file.path,
                        "size": file.size,
                        "blob_id": file.blob_id,
                        "lfs_oid": file.lfs_oid,
                        "status": status,
                    }
                )
            results["models"].append(model_result)

    write_lock_file(output_dir, results)
    return results


def write_lock_file(output_dir: Path, data: dict[str, Any]) -> None:
    lock_path = output_dir / "model-download-lock.json"
    tmp_path = output_dir / "model-download-lock.json.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, lock_path)


def main() -> int:
    args = parse_args()
    if args.download and (args.verify_only or args.manifest_only):
        raise SystemExit("--download cannot be combined with --verify-only or --manifest-only")

    needs_remote = not args.list and not args.manifest_only
    deps = load_runtime_deps(require_hf=needs_remote, require_requests=args.download)
    load_dotenv(args, deps)
    raw_manifest = read_manifest(args.manifest, deps["yaml"])
    configs = build_model_configs(raw_manifest)

    if args.list:
        print_manifest_list(deps, active_configs(configs, args))
        return 0

    selected = select_models(configs, args)
    if args.manifest_only:
        print_manifest_sizes(deps, selected)
        return 0

    output_dir = resolve_output_dir(raw_manifest, args)
    token = resolve_hf_token(args)
    plans = resolve_plans(deps, selected, output_dir, token, args.revision)
    enforce_size_limits(deps, plans, raw_manifest, args)
    print_remote_plan(deps, plans, output_dir)

    if args.verify_only:
        failures = verify_local_files(deps, plans)
        return 1 if failures else 0

    if not args.download:
        deps["Console"]().print("Run with [bold]--download[/bold] to fetch the model files.")
        return 0

    check_disk_space(deps, plans, output_dir, raw_manifest, args)
    download_plans(deps, plans, output_dir, token, args)
    deps["Console"]().print(f"[green]Done.[/green] Lock file written to {output_dir / 'model-download-lock.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
