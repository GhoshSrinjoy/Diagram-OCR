"""Command line for the Diagram IR.

Usage (from the `src/` directory, or with `src` on PYTHONPATH):
    python -m diagram_ir convert flow.dgm --to json            # print JSON
    python -m diagram_ir convert flow.dgm -o flow.xml          # format from the extension
    python -m diagram_ir convert flow.json --to dgm --canonical
    python -m diagram_ir convert out.dgm --to json,yaml,xml -o exports/   # several at once
    python -m diagram_ir validate flow.yaml --strict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .canonical import canonicalize
from .convert import EXTENSIONS, FORMATS, detect_format, dump, load
from .dsl import DgmSyntaxError
from .model import errors, validate

_OUT_EXT = {"dgm": ".dgm", "json": ".json", "yaml": ".yaml", "xml": ".xml"}


def _read(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def _src_format(path: str, text: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return EXTENSIONS.get(Path(path).suffix.lower()) or detect_format(text)


def cmd_convert(args: argparse.Namespace) -> int:
    text = _read(args.input)
    d = load(text, _src_format(args.input, text, args.src))
    if args.canonical:
        d = canonicalize(d)

    targets = [t.strip() for t in args.to.split(",")] if args.to else []
    out = Path(args.output) if args.output else None
    if not targets:
        if out is None or out.suffix.lower() not in EXTENSIONS:
            print("error: give --to, or an -o path with a known extension", file=sys.stderr)
            return 2
        targets = [EXTENSIONS[out.suffix.lower()]]
    for t in targets:
        if t not in FORMATS:
            print(f"error: unknown format {t!r}; expected one of {FORMATS}", file=sys.stderr)
            return 2

    if len(targets) > 1 and out is not None:  # -o is a directory
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem if args.input != "-" else "diagram"
        for t in targets:
            (out / f"{stem}{_OUT_EXT[t]}").write_text(dump(d, t), encoding="utf-8")
        return 0
    for t in targets:
        rendered = dump(d, t)
        if out is not None:
            out.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    text = _read(args.input)
    d = load(text, _src_format(args.input, text, args.src))
    issues = validate(d, strict=args.strict)
    for issue in issues:
        print(issue)
    n_err = len(errors(issues))
    print(f"{len(d.nodes)} nodes, {len(d.edges)} edges, {len(d.groups)} groups; "
          f"{n_err} errors, {len(issues) - n_err} warnings")
    return 1 if n_err else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m diagram_ir", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert", help="convert between dgm / json / yaml / xml")
    p.add_argument("input", help="input file, or - for stdin")
    p.add_argument("--from", dest="src", choices=FORMATS, help="input format (default: detect)")
    p.add_argument("--to", help="output format(s), comma-separated")
    p.add_argument("-o", "--output", help="output file (or directory for several formats)")
    p.add_argument("--canonical", action="store_true", help="renumber ids / reorder (training form)")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("validate", help="check structure and vocabulary")
    p.add_argument("input")
    p.add_argument("--from", dest="src", choices=FORMATS)
    p.add_argument("--strict", action="store_true", help="vocabulary mismatches are errors")
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DgmSyntaxError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
