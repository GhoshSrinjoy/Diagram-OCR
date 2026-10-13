"""Batch-render PlantUML sources to PNG with one JVM per batch.

render_batch(items, out_dir) takes (key, code) pairs and returns {key: png_path} for the ones that
rendered cleanly. Files PlantUML reports errors for are dropped (it still writes an error image),
as are images that hit the size limit (PlantUML crops them, so they would not match the target).
Only the first page of a file is used, matching the importers.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

PLANTUML_JAR = Path("D:/data/tools/plantuml.jar")
LIMIT = 6000  # px; PlantUML's default 4096 crops large diagrams
_ERROR = re.compile(r"Error line \d+ in file: (.+)$", re.MULTILINE)
_OLD_HANDWRITTEN = re.compile(r"(?im)^[ \t]*skinparam[ \t]+handwritten[ \t]+true[ \t]*$")
_BLOCK = re.compile(r"^[ \t]*@startuml\b[^\n]*\n(.*?)^[ \t]*@enduml\b",
                    re.MULTILINE | re.DOTALL | re.IGNORECASE)


def first_block(code: str) -> str:
    """The first @startuml..@enduml block with a bare header: `@startuml Name` would rename the
    output file, and files without the markers are ignored by PlantUML (the importer reads the
    same first block, so image and target stay in sync)."""
    m = _BLOCK.search(code)
    body = m.group(1) if m else re.sub(r"(?im)^[ \t]*@(start|end)uml\b.*$", "", code)
    # PlantUML 1.2026 draws a deprecation banner for the old spelling - text that is in the
    # image but not in the target. The current spelling renders the same sketch style cleanly.
    body = _OLD_HANDWRITTEN.sub("!option handwritten true", body)
    return "@startuml\n" + body.rstrip() + "\n@enduml\n"


def render_batch(
    items: list[tuple[str, str]], out_dir: Path, threads: int = 6, timeout: int = 1800
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    done: dict[str, Path] = {}
    todo = []
    for key, code in items:
        dest = out_dir / f"{key}.png"
        if dest.exists() and dest.stat().st_size > 0:
            done[key] = dest
        else:
            todo.append((key, code))
    if not todo:
        return done
    with tempfile.TemporaryDirectory(prefix="puml_") as tmp:
        src, png = Path(tmp) / "src", Path(tmp) / "png"
        src.mkdir()
        for key, code in todo:
            (src / f"{key}.puml").write_text(first_block(code), encoding="utf-8")
        cmd = ["java", "-Djava.awt.headless=true", f"-DPLANTUML_LIMIT_SIZE={LIMIT}", "-Xmx4g",
               "-jar", str(PLANTUML_JAR), "-tpng", "-charset", "UTF-8", "-nbthread", str(threads),
               "-o", str(png), str(src / "*.puml")]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                              errors="replace")
        failed = {Path(m.strip()).stem for m in _ERROR.findall(proc.stdout + proc.stderr)}
        for key, _ in todo:
            first = png / f"{key}.png"
            if key in failed or not first.exists():
                continue
            try:
                with Image.open(first) as im:
                    w, h = im.size
            except OSError:
                continue
            if w >= LIMIT or h >= LIMIT or w < 16 or h < 16:
                continue
            dest = out_dir / f"{key}.png"
            shutil.move(str(first), dest)
            done[key] = dest
    return done
