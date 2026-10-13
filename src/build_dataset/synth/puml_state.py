"""Synthetic PlantUML state diagrams: simple states with description lines, composite states (nested
sometimes), concurrent regions (`--` / `||`), one [*] initial / final per scope, choice / fork /
join pseudo-states, history, transitions labelled `event [guard] / action`, and notes.
`generate` returns the PlantUML body and the Diagram the importer should produce for it (see
puml_check): composites are groups, transitions into / out of a composite end on the group, a
region's nodes belong to its region group."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from diagram_ir import Diagram, Group

from .puml_check import Intent
from .puml_words import Domain, camel, ident, is_keyword, words

GENERIC_STATES = ("Idle", "Ready", "Waiting", "Done", "Failed", "Retrying", "Paused", "Starting", "Stopping",
                  "Error", "Init", "Validating", "Submitted", "Archived", "Locked", "Queued", "Scheduled",
                  "Expired", "Suspended", "Completed")
COMPOSITE_NAMES = ("Active", "Processing", "Operational", "Running", "Session", "InService", "Online",
                   "Working",
                   "Handling", "Open", "Busy", "Enabled")
DESCRIPTIONS = ("entry / start timer", "exit / stop timer", "do / poll sensor", "entry / log state",
                "do / wait for input", "exit / save draft", "entry / notify owner", "do / retry request",
                "timeout 30s", "shows spinner", "awaiting response", "read-only", "entry / lock record",
                "exit / release lock")
NOTE_TEXTS = ("terminal state", "may be retried", "kept for 30 days", "requires approval", "set by scheduler",
              "visible to customer", "audit logged", "auto-closes after 7 days", "manual intervention",
              "default state")


@dataclass(eq=False)
class _Scope:
    group: Optional[Group]  # None at top level
    states: list[str] = field(default_factory=list)  # state names declared here
    initial: Optional[str] = None
    final: Optional[str] = None


class _Gen:
    def __init__(self, rng: random.Random, dom: Domain):
        self.rng = rng
        self.dom = dom
        self.it = Intent("state", "TB")
        self.out: list[str] = []
        self.taken: set[str] = set()
        self.nid: dict[str, str] = {}  # state name -> node id (or group id for composites)
        self.code: dict[str, str] = {}  # state name -> code reference
        self.names = list(dom.states)
        rng.shuffle(self.names)
        extra = list(GENERIC_STATES)
        rng.shuffle(extra)
        self.names += extra  # domain names first
        self.events = list(dom.events)
        rng.shuffle(self.events)
        self.notes = list(NOTE_TEXTS)
        rng.shuffle(self.notes)
        self.n_pseudo = 0
        self.indent = ""

    def name(self, pool: Optional[list[str]] = None) -> Optional[str]:
        for cand in pool if pool is not None else self.names:
            ref = cand if cand.isidentifier() else ident(cand)
            if cand.lower() in self.taken or ref.lower() in self.taken or is_keyword(cand) or is_keyword(ref):
                continue
            self.taken.update({cand.lower(), ref.lower()})
            self.code[cand] = ref
            return cand
        return None

    def emit(self, line: str) -> None:
        self.out.append(self.indent + line)

    def member(self, scope: _Scope, nid: str) -> None:
        if scope.group is not None:
            scope.group.members.append(nid)

    def label(self) -> str:
        rng = self.rng
        ev = self.events.pop() if self.events else rng.choice(self.dom.events)
        guard = f"[{rng.choice(self.dom.checks).rstrip('?').lower()}]"
        act = f"/ {camel(words(rng.choice(self.dom.verbs)))}()" if rng.random() < 0.6 else \
            f"/ {rng.choice(['log', 'notify', 'save', 'reset counter', 'send email'])}"
        r = rng.random()
        if r < 0.55:
            return ev
        if r < 0.7:
            return f"{ev} {guard}"
        if r < 0.8:
            return f"{ev} {act}"
        if r < 0.85:
            return f"{ev} {guard} {act}"
        if r < 0.9:
            return guard
        return ""

    def arrow(self, labelled: bool) -> str:
        """Graphviz drops some labels of same-rank links (`->`, left / right hints: ~2% measured), so
        a labelled transition only gets a two-dash spelling."""
        r = self.rng.random()
        if labelled:
            return "-[dashed]->" if r >= 0.97 else "-->"
        if r < 0.6:
            return "-->"
        if r < 0.75:
            return "->"
        if r < 0.97:
            return f"-{self.rng.choice(['down', 'right', 'left', 'up', 'd', 'r', 'l', 'u'])}->"
        return "-[dashed]->"

    def transition(self, a_code: str, b_code: str, a_id: str, b_id: str, label: Optional[str] = None) -> None:
        label = self.label() if label is None else label
        arrow = self.arrow(bool(label))
        style = "dashed" if "dashed" in arrow else None
        self.emit(f"{a_code} {arrow} {b_code}" + (f" : {label}" if label else ""))
        self.it.edge(a_id, b_id, "transition", label, style=style)

    def start(self, scope: _Scope) -> str:
        if scope.initial is None:
            scope.initial = self.it.node("initial")
            self.member(scope, scope.initial)
        return scope.initial

    def end(self, scope: _Scope) -> str:
        if scope.final is None:
            scope.final = self.it.node("final")
            self.member(scope, scope.final)
        return scope.final

    def simple_states(self, scope: _Scope, k: int) -> list[str]:
        """Declare k simple states (some only by use) in `scope`; -> their names."""
        rng = self.rng
        out = []
        for _ in range(k):
            name = self.name()
            if name is None:
                break
            ref = self.code[name]
            desc: list[str] = []
            if rng.random() < 0.25:
                desc = rng.sample(DESCRIPTIONS, rng.choice([1, 1, 2]))
            nid = self.it.node("state", name, members=[("attr", d) for d in desc])
            self.nid[name] = nid
            self.member(scope, nid)
            scope.states.append(name)
            out.append(name)
            quoted = ref != name
            r = rng.random()
            if quoted:
                self.emit(f'state "{name}" as {ref}' + (f" : {desc[0]}" if desc and r < 0.5 else ""))
                rest = desc[1:] if desc and r < 0.5 else desc
            elif desc and r < 0.4:
                self.emit(f"state {ref} : {desc[0]}")
                rest = desc[1:]
            elif r < 0.5:
                self.emit(f"state {ref}")
                rest = desc
            else:  # declared by its first use, unless it has description lines
                rest = desc
            for d in rest:
                self.emit(f"{ref} : {d}")
        return out

    def wire(self, scope: _Scope, names: list[str], final_p: float) -> None:
        """[*] -> first, a chain with some extra / back edges and self loops, maybe -> [*]."""
        rng = self.rng
        if not names:
            return
        self.transition("[*]", self.code[names[0]], self.start(scope), self.nid[names[0]],
                        label="" if rng.random() < 0.8 else None)
        for a, b in zip(names, names[1:], strict=False):
            self.transition(self.code[a], self.code[b], self.nid[a], self.nid[b])
        for _ in range(rng.randint(0, max(0, len(names) - 1))):
            a, b = rng.choice(names), rng.choice(names)
            if a == b and rng.random() < 0.7:
                continue
            self.transition(self.code[a], self.code[b], self.nid[a], self.nid[b])
        if rng.random() < final_p:
            for a in rng.sample(names, rng.choice([1, 1, 2]) if len(names) > 1 else 1):
                self.transition(self.code[a], "[*]", self.nid[a], self.end(scope),
                                label="" if rng.random() < 0.6 else None)

    def composite(self, parent: _Scope, depth: int) -> Optional[str]:
        rng = self.rng
        pool = list(COMPOSITE_NAMES) + self.names
        rng.shuffle(pool)
        name = self.name(pool)
        if name is None:
            return None
        ref = self.code[name]
        g = self.it.group("composite", name, parent.group.id if parent.group else None)
        self.nid[name] = g.id
        head = f'state "{name}" as {ref} {{' if ref != name else f"state {ref} {{"
        self.emit(head)
        old = self.indent
        self.indent += "  "
        if depth < 2 and rng.random() < 0.15:  # concurrent regions
            sep = rng.choice(["--", "||"])
            for k in range(2):
                if k:
                    self.emit(sep)
                region = self.it.group("region", "", g.id)
                sc = _Scope(region)
                names = self.simple_states(sc, rng.randint(1, 3))
                self.wire(sc, names, 0.5)
        else:
            sc = _Scope(g)
            names = self.simple_states(sc, rng.randint(2, 4))
            if names and depth < 1 and rng.random() < 0.12:
                inner = self.composite(sc, depth + 1)
                if inner:
                    last = names[-1]
                    self.transition(self.code[last], self.code[inner], self.nid[last], self.nid[inner])
            self.wire(sc, names, 0.55)
        self.indent = old
        self.emit("}")
        return name

    def pseudo(self, kind: str) -> tuple[str, str]:
        self.n_pseudo += 1
        stem = self.rng.choice(["check", "decide", "c", "choice"]) if kind == "choice" else kind
        code = f"{stem}{self.n_pseudo}"
        self.emit(f"state {code} <<{kind}>>")
        return code, self.it.node(kind)


def generate(rng: random.Random, dom: Domain, direction: Optional[str] = "TB",
             colors: bool = True) -> tuple[list[str], Diagram]:
    g = _Gen(rng, dom)
    g.it.d.direction = direction
    top = _Scope(None)
    n_simple = rng.choices(range(2, 9), weights=(14, 20, 20, 16, 12, 10, 8))[0]
    names = g.simple_states(top, n_simple)
    comps: list[str] = []
    for _ in range(rng.choices([0, 1, 2], weights=(55, 35, 10))[0]):
        c = g.composite(top, 0)
        if c:
            comps.append(c)
    g.wire(top, names, 0.7)
    every = names + comps
    for c in comps:  # enter and leave each composite
        a = rng.choice(names)
        g.transition(g.code[a], g.code[c], g.nid[a], g.nid[c])
        if rng.random() < 0.8:
            b = rng.choice(names)
            g.transition(g.code[c], g.code[b], g.nid[c], g.nid[b])
    if len(every) >= 3 and rng.random() < 0.22:  # a choice point
        code, nid = g.pseudo("choice")
        a = rng.choice(every)
        g.transition(g.code[a], code, g.nid[a], nid, label="" if rng.random() < 0.7 else None)
        others = [x for x in every if x != a] or every
        for b in rng.sample(others, min(len(every) - 1, rng.choice([2, 2, 3]))):
            guard = f"[{rng.choice(dom.checks).rstrip('?').lower()}]"
            g.transition(code, g.code[b], nid, g.nid[b], label=guard if rng.random() < 0.85 else "")
    if len(names) >= 4 and rng.random() < 0.1:  # fork / join
        fcode, fid = g.pseudo("fork")
        jcode, jid = g.pseudo("join")
        a, b, c, d = rng.sample(names, 4)
        g.transition(g.code[a], fcode, g.nid[a], fid, label="")
        for x in (b, c):
            g.transition(fcode, g.code[x], fid, g.nid[x], label="")
            g.transition(g.code[x], jcode, g.nid[x], jid, label="")
        g.transition(jcode, g.code[d], jid, g.nid[d], label="")
    if comps and rng.random() < 0.08:  # re-enter a composite through its history
        c = rng.choice(comps)
        deep = rng.random() < 0.25
        hist = g.it.node("deep_history" if deep else "history")
        grp = next(x for x in g.it.d.groups if x.id == g.nid[c])
        if not any(x.parent == grp.id and x.kind == "region" for x in g.it.d.groups):
            grp.members.append(hist)
            a = rng.choice(names)
            g.transition(g.code[a], f"{g.code[c]}[{'H*' if deep else 'H'}]", g.nid[a], hist)
        else:
            g.it.d.nodes.pop()
    for _ in range(rng.choices([0, 1, 2], weights=(75, 18, 7))[0]):
        if not g.notes:
            break
        s = rng.choice(names)
        text = g.notes.pop()
        g.emit(f"note {rng.choice(['left', 'right', 'top', 'bottom'])} of {g.code[s]} : {text}")
        g.it.edge(g.it.node("note", text), g.nid[s], "anchor")
    return g.out, g.it.d
