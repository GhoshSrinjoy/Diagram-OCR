"""Synthetic PlantUML diagrams (generated in-house) -> rendered UML samples.

Generators in build_dataset/synth/puml_*.py write random but realistic PlantUML for class, sequence,
activity (about half with swimlanes - the BPMN-like case), state, use case and component /
deployment diagrams, with names from 14 business domains, and a random look (puml_style: built-in
themes on an opaque page, `!option handwritten true`, handwriting and plain fonts, monochrome,
ortho / polyline lines, left-to-right layout, colours, per-kind skinparams).

Targets come from the same importer as the real-world UML data (plantuml_to_diagram on the first
@startuml block, exactly what render_batch draws), so conventions match. Each generator also builds
the Diagram it means to draw; a sample is kept only if the importer's result equals it up to ids and
order (puml_check.compare: nodes, edges with labels / end labels / style, groups, sequence order).
Mismatches are skipped and reported by category - they point at importer or generator bugs.

Image <-> target text gate: each batch is also drawn as SVG (same jar and flags) and a sample is kept
only if every word of the target (node / member / label / multiplicity / group text) is drawn and
nothing else is (apart from sequence fragment keywords, autonumber digits and the history `H`).
It also drops renders where Graphviz drew an element inside a container it is not a member of.
This catches renderer quirks the source cannot show: Graphviz dropping an edge label it cannot
place or misplacing a node, deprecation banners, labels a style does not draw, crash images.

Renders: one JVM per batch (render_plantuml.render_batch, <= 6 threads); image names carry a hash
of the source so a changed generator never reuses a stale image. Splits by hash of the sample id:
3% test, 3% val, the rest train.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Optional

from diagram_ir import Diagram, canonicalize, dumps, validate
from diagram_ir.importers.plantuml import plantuml_to_diagram

from ..common import OUT_ROOT, Sample, SkipLog, make_sample
from ..render_plantuml import first_block, render_batch
from ..synth import puml_activity, puml_class, puml_component, puml_sequence, puml_state, puml_usecase
from ..synth.puml_check import compare
from ..synth.puml_style import make_style
from ..synth.puml_svg import drawn_words, misplaced, render_svgs, words
from ..synth.puml_words import pick_domain
from ._plantuml_sets import MAX_TARGET_CHARS

SOURCE = "synth_plantuml"
LICENSE = "synthetic (generated in-house)"
QUOTA = {"class": 3000, "sequence": 2500, "activity": 2000, "state": 1200, "usecase": 800,
         "component": 250, "deployment": 250}
SEED = "synth_plantuml/v1"
BATCH = 500
THREADS = 6  # two other generators share the machine
MAX_TRIES = 3  # candidates per kind and quota slot before giving up on that kind

GENERATORS: dict[str, Callable[..., tuple[list[str], Diagram]]] = {
    "class": puml_class.generate,
    "sequence": puml_sequence.generate,
    "activity": puml_activity.generate,
    "state": puml_state.generate,
    "usecase": puml_usecase.generate,
    "component": puml_component.generate,
    "deployment": puml_component.generate,
}


def split_of(sample_id: str) -> str:
    h = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < 0.03 else ("val" if h < 0.06 else "train")


def generate(kind: str, index: int) -> tuple[str, Diagram, dict]:
    """(PlantUML source as rendered, intended Diagram, sample meta) for one candidate."""
    rng = random.Random(f"{SEED}:{kind}:{index}")
    dom = pick_domain(rng)
    lanes = rng.random() < 0.5 if kind == "activity" else False
    style = make_style(rng, kind, has_lanes=lanes)
    kw: dict = {"lanes": lanes} if kind == "activity" else {}
    if kind in ("component", "deployment"):
        kw["flavor"] = kind
    direction = style.direction or (None if kind == "sequence" else "TB")
    body, intent = GENERATORS[kind](rng, dom, direction=direction, colors=style.theme == "default", **kw)
    code = first_block("@startuml\n" + "\n".join(style.lines + body) + "\n@enduml\n")
    meta = {"diagram_type": kind, "theme": style.theme, "handwritten": style.handwritten, "domain": dom.key}
    if kind == "activity":
        meta["swimlanes"] = any(g.kind == "lane" for g in intent.groups)
    return code, intent, meta


def check(kind: str, code: str, intent: Diagram) -> tuple[Optional[Diagram], str, str]:
    """(importer result, "", "") if it matches the intent and makes a valid target, else
    (None, reason, detail)."""
    try:
        d = plantuml_to_diagram(code)
    except Exception as exc:  # the importer raises PlantUMLError subclasses; be robust to the rest
        return None, f"{kind}: importer error {type(exc).__name__}", str(exc)[:120]
    diff = compare(intent, d)
    if diff is not None:
        return None, f"{kind}: importer != generator ({diff[0]})", diff[1]
    canon = canonicalize(d)
    issues = validate(d, strict=True) + validate(canon, strict=True)
    if issues:
        return None, f"{kind}: target not strictly valid", str(issues[0])
    if len(dumps(canon)) > MAX_TARGET_CHARS:
        return None, f"{kind}: target too long", f"{len(dumps(canon))} chars"
    return d, "", ""


_FRAGMENT_TABS = frozenset({"alt", "opt", "loop", "par", "break", "critical", "group", "ref"})


def render_problem(d: Diagram, code: str, svg: Optional[str]) -> tuple[str, str]:
    """("", "") if the SVG render shows exactly the target, else (reason, detail)."""
    if svg is None:
        return "no SVG (render error)", ""
    drawn = drawn_words(svg)
    want: set[str] = set()
    for n in d.nodes:
        want |= words(n.text) | words(n.stereotype or "")
        for m in n.members:
            want |= words(m.text)
    for e in d.edges:
        want |= words(e.label) | words(e.source_label or "") | words(e.target_label or "")
    for g in d.groups:
        want |= words(g.text)
    missing = want - drawn
    if missing:
        return "target text not drawn", " ".join(sorted(missing)[:5])
    extra = {w for w in drawn - want - _FRAGMENT_TABS
             if not (w.isdigit() and "autonumber" in code) and not (w == "h" and "[H" in code)}
    if extra:
        return "drawn text not in target", " ".join(sorted(extra)[:5])
    wrong = misplaced(svg)
    if wrong:
        return "element drawn inside a container it is not in", f"{wrong[0][0]} in {wrong[0][1]}"
    return "", ""


def _schedule() -> list[str]:
    """All quota slots, kinds interleaved in proportion (a --limit run gets every kind)."""
    slots = sorted(((i + 0.5) / q, kind) for kind, q in QUOTA.items() for i in range(q))
    return [kind for _f, kind in slots]


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    out_dir = OUT_ROOT / "images" / SOURCE
    made: Counter = Counter()
    tried: Counter = Counter()
    nxt: Counter = Counter()  # next candidate index per kind
    t0 = time.time()
    todo = _schedule()
    while todo:
        chunk, todo = todo[:BATCH], todo[BATCH:]
        batch: list[tuple[str, str, str, Diagram, dict]] = []  # (kind, local id, code, diagram, meta)
        for kind in chunk:
            if made[kind] + sum(b[0] == kind for b in batch) >= QUOTA[kind]:
                continue
            for _ in range(MAX_TRIES):
                index = nxt[kind]
                nxt[kind] += 1
                tried[kind] += 1
                code, intent, meta = generate(kind, index)
                d, reason, detail = check(kind, code, intent)
                if d is not None:
                    batch.append((kind, f"{kind}_{index:05d}", code, d, meta))
                    break
                skips.skip(reason, f"{kind}_{index:05d}: {detail}")
        keys = {lid: f"{lid}_{hashlib.md5(code.encode('utf-8')).hexdigest()[:6]}"
                for _k, lid, code, _d, _m in batch}
        svgs = render_svgs([(keys[lid], code) for _k, lid, code, _d, _m in batch], THREADS)
        checked = []
        for kind, lid, code, d, meta in batch:
            reason, detail = render_problem(d, code, svgs.get(keys[lid]))
            if reason:
                skips.skip(f"{kind}: {reason}", f"{lid}: {detail}")
                todo.append(kind)  # try another candidate for this slot
                continue
            checked.append((kind, lid, code, d, meta))
        batch = checked
        jobs = [(keys[lid], code) for _k, lid, code, _d, _m in batch]
        rendered = render_batch(jobs, out_dir, threads=THREADS)
        for kind, lid, _code, d, meta in batch:
            png = rendered.get(keys[lid])
            if png is None:
                skips.skip(f"{kind}: render failed / cropped", lid)
                todo.append(kind)  # try another candidate for this slot
                continue
            sid = f"{SOURCE}/{lid}"
            try:
                sample = make_sample(source=SOURCE, local_id=lid, split=split_of(sid), image=png, diagram=d,
                                     use="train", license=LICENSE, meta=meta)
            except ValueError as exc:
                skips.skip(f"{kind}: invalid diagram", f"{lid}: {exc}")
                continue
            made[kind] += 1
            yield sample
        print(f"  {SOURCE}: {sum(made.values())} samples, {sum(tried.values())} candidates "
              f"({time.time() - t0:.0f}s)", flush=True)
        if todo and all(tried[k] >= MAX_TRIES * QUOTA[k] for k in set(todo)):
            break  # every remaining kind keeps failing: stop instead of looping
    print(f"  {SOURCE}: made {dict(made)} from {dict(tried)} candidates in {time.time() - t0:.0f}s")
    skips.report()
