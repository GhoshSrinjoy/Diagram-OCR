"""Synthetic PlantUML use case diagrams: actors (all declaration spellings), use cases inside one or
two system boundaries, actor associations, <<include>> / <<extend>>, actor and use case
generalization, notes. The importer maps them to the generic `graph` type (actor -> actor, use case
-> ellipse, boundary -> cluster). `generate` returns the PlantUML body and that Diagram."""

from __future__ import annotations

import random
from typing import Optional

from diagram_ir import Diagram

from .puml_check import Intent
from .puml_words import Domain, ident, is_keyword

NOTE_TEXTS = ("requires login", "max 3 attempts", "via mobile app", "only on weekdays", "logged for audit",
              "needs manager approval", "optional", "uses 2FA", "async", "see business rule BR-7")


class _El:
    def __init__(self, text: str, code: str, kind: str, decl: str):
        self.text, self.code, self.kind, self.decl = text, code, kind, decl
        self.nid = ""


def _actor(rng: random.Random, text: str, k: int, taken: set[str]) -> _El:
    simple = text.isidentifier() and not is_keyword(text)
    r = rng.random()
    if simple and r < 0.45:
        return _El(text, text, "actor", f"actor {text}")
    if simple and r < 0.65:
        return _El(text, text, "actor", f":{text}:")
    code = ident(text) if rng.random() < 0.6 else f"A{k}"
    if code.lower() in taken:
        code = f"A{k}x"
    if r < 0.85:
        return _El(text, code, "actor", f'actor "{text}" as {code}')
    return _El(text, code, "actor", f":{text}: as {code}")


def _usecase(rng: random.Random, text: str, k: int) -> _El:
    r = rng.random()
    code = f"UC{k}" if rng.random() < 0.7 else ident(text)
    if r < 0.45:
        return _El(text, code, "ellipse", f'usecase "{text}" as {code}')
    if r < 0.7:
        return _El(text, f"({text})", "ellipse", f"({text})")
    if r < 0.88:
        return _El(text, code, "ellipse", f"({text}) as {code}")
    return _El(text, code, "ellipse", f'usecase {code} as "{text}"')


def _link(rng: random.Random, it: Intent, a: _El, b: _El, kind: str, label: str = "") -> str:
    """kind: assoc (actor - use case), include / extend (dashed arrow a -> b), general (a --|> b)."""
    if kind == "assoc":
        r = rng.random()
        if r < 0.45:
            it.edge(a.nid, b.nid, "arrow")
            return f"{a.code} {rng.choice(['-->', '-->', '->', '--->'])} {b.code}"
        if r < 0.55:
            it.edge(a.nid, b.nid, "arrow")
            return f"{b.code} <-- {a.code}"
        it.edge(a.nid, b.nid, "line")
        return f"{a.code} {rng.choice(['--', '--', '-', '---'])} {b.code}"
    if kind in ("include", "extend"):
        it.edge(a.nid, b.nid, "arrow", f"«{kind}»", style="dashed")
        arrow = rng.choice(["..>", "..>", "...>"])  # no `.>`: same-rank link labels can be dropped
        if rng.random() < 0.15:
            return f"{b.code} <{arrow[:-1]} {a.code} : <<{kind}>>"
        return f"{a.code} {arrow} {b.code} : <<{kind}>>"
    it.edge(a.nid, b.nid, "arrow")  # generalization: the hollow triangle is an arrow head here
    if rng.random() < 0.3:
        return f"{b.code} <|-- {a.code}"
    return f"{a.code} {rng.choice(['--|>', '-|>'])} {b.code}"


def generate(rng: random.Random, dom: Domain, direction: Optional[str] = "TB",
             colors: bool = True) -> tuple[list[str], Diagram]:
    it = Intent("graph", direction)
    taken: set[str] = set()
    n_act = rng.choices([1, 2, 3, 4, 5], weights=(18, 32, 26, 15, 9))[0]
    n_uc = rng.choices(range(3, 13), weights=(10, 14, 15, 14, 12, 10, 8, 7, 5, 5))[0]
    actor_names = [x for x in dom.actors if x.lower() not in taken]
    rng.shuffle(actor_names)
    actors = []
    for k, text in enumerate(actor_names[:n_act], 1):
        a = _actor(rng, text, k, taken)
        taken.update({a.text.lower(), a.code.lower()})
        actors.append(a)
    uc_names = [x for x in dom.usecases if x.lower() not in taken]
    rng.shuffle(uc_names)
    ucs = []
    for k, text in enumerate(uc_names[:n_uc], 1):
        u = _usecase(rng, text, k)
        if u.code.lower() in taken:
            u = _El(text, f"({text})", "ellipse", f"({text})")
        taken.update({u.text.lower(), u.code.lower()})
        ucs.append(u)
    for a in actors:
        a.nid = it.node("actor", a.text)
    for u in ucs:
        u.nid = it.node("ellipse", u.text)

    # ---- boundaries ----------------------------------------------------------------
    lines: list[str] = []
    boxes: list[tuple[str, list[_El]]] = []
    r = rng.random()
    if r < 0.7 and len(ucs) >= 2:
        systems = list(dom.systems)
        rng.shuffle(systems)
        if r < 0.12 and len(ucs) >= 5:
            cut = rng.randint(2, len(ucs) - 2)
            boxes = [(systems[0], ucs[:cut]), (systems[1], ucs[cut:])]
        else:
            inside = ucs if rng.random() < 0.8 else ucs[:max(2, len(ucs) - rng.randint(1, 2))]
            boxes = [(systems[0], inside)]
    boxed = {id(u) for _t, us in boxes for u in us}
    actors_first = rng.random() < 0.6
    if actors_first:
        lines += [a.decl for a in actors]
    for title, us in boxes:
        kw = rng.choices(["rectangle", "package", "frame"], weights=(70, 20, 10))[0]
        lines.append(f'{kw} "{title}" {{')
        g = it.group("cluster", title)
        for u in us:
            lines.append("  " + u.decl)
            g.members.append(u.nid)
        lines.append("}")
    lines += [u.decl for u in ucs if id(u) not in boxed]
    if not actors_first:
        lines += [a.decl for a in actors]

    # ---- links -----------------------------------------------------------------------
    links: list[str] = []
    used: set[tuple[int, int]] = set()

    def link(a: _El, b: _El, kind: str) -> None:
        key = (min(id(a), id(b)), max(id(a), id(b)))
        if a is b or key in used:
            return
        used.add(key)
        links.append(_link(rng, it, a, b, kind))

    for a in actors:
        for u in rng.sample(ucs, min(len(ucs), rng.choice([1, 2, 2, 3, 4]))):
            link(a, u, "assoc")
    orphans = [u for u in ucs if not any(id(u) in k for k in used)]
    for u in orphans:
        if rng.random() < 0.5 and len(ucs) > 1:
            base = rng.choice([x for x in ucs if x is not u])
            if rng.random() < 0.6:
                link(base, u, "include")
            else:
                link(u, base, "extend")
        else:
            link(rng.choice(actors), u, "assoc")
    for _ in range(rng.choices([0, 1, 2], weights=(55, 30, 15))[0]):
        if len(ucs) >= 2:
            a, b = rng.sample(ucs, 2)
            link(a, b, rng.choice(["include", "extend"]))
    if len(actors) >= 2 and rng.random() < 0.18:
        child, parent = rng.sample(actors, 2)
        link(child, parent, "general")
    if len(ucs) >= 3 and rng.random() < 0.08:
        child, parent = rng.sample(ucs, 2)
        link(child, parent, "general")
    rng.shuffle(links)
    lines += links

    # ---- notes -----------------------------------------------------------------------
    if rng.random() < 0.12:
        u = rng.choice(ucs)
        text = rng.choice(NOTE_TEXTS)
        nid = it.node("note", text)
        if rng.random() < 0.7:
            lines.append(f"note {rng.choice(['left', 'right', 'top', 'bottom'])} of {u.code} : {text}")
        else:
            lines.append(f'note "{text}" as N1')
            lines.append(f"N1 .. {u.code}")
        it.edge(nid, u.nid, "anchor")
    return lines, it.d
