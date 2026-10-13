"""Random block-structured BPMN processes (always sound) for the synthetic BPMN source.

A process is a tree of blocks: sequences, gateway blocks (XOR / AND / OR split + merge,
event-based gateway + XOR merge), loops (XOR merge ... XOR split with a back edge) and early
exits (XOR split whose side branch ends in an end event; the rest of the sequence continues as
the other branch). The tree is grown by random refinements until it has the requested number of
flow nodes, then decorated with business labels, event types and task markers.

The same tree drives the layout (bpmn_layout): every block is drawn as a rectangle of grid cells,
so the drawing is planar by construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Union

from . import bpmn_vocab as V

FLOW_KINDS = ("task", "start", "end", "inter", "gw")


@dataclass
class El:
    """One drawn BPMN element."""

    id: str
    kind: str  # task start end inter gw data store note
    gw: str = ""  # gateway type: xor and or evgw
    split: bool = False  # gateway role: fan-out (True) or fan-in (False)
    ev: str = "none"  # event definition: none message timer terminate signal conditional escalation
    throw: bool = False  # intermediate event: throwing (True) or catching
    marker: str = ""  # task: sub loop mi mis; xor gateway: blank
    icon: str = ""  # task type icon (clean style only): user service send receive script rule
    text: str = ""
    pool: str = "main"  # process / pool key
    # layout, in layout units (filled by bpmn_layout)
    col: int = 0
    row: int = 0
    x: float = 0.0  # centre
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    lines: list[str] = field(default_factory=list)  # drawn text lines (internal or external label)
    label_box: Optional[tuple[float, float, float, float]] = None  # external label (events, ...)


@dataclass
class Flow:
    src: str
    dst: str
    kind: str = "seq"  # seq msg data assoc
    label: str = ""
    hint: str = "seq"  # routing: seq split join back skip
    side: int = 1  # corridor / exit side: +1 below, -1 above
    corridor: int = 0  # relative corridor row id (back / skip edges), resolved by the layout
    points: list[tuple[float, float]] = field(default_factory=list)
    label_lines: list[str] = field(default_factory=list)
    label_box: Optional[tuple[float, float, float, float]] = None


@dataclass
class Leaf:
    el: El


@dataclass
class Seq:
    items: list = field(default_factory=list)
    lock_first: bool = False  # event-based gateway branch: the catch event must stay first


@dataclass
class Split:
    split: El
    merge: Optional[El]
    branches: list[Seq]
    labels: list[str]
    term: list[bool]  # branch ends in an end event (no flow into the merge)
    order: list[int] = field(default_factory=list)  # vertical placement: offsets of branches


@dataclass
class Loop:
    merge: Optional[El]  # None: the back edge goes straight into the first body task
    split: El
    body: Seq
    fwd: str = ""
    back: str = ""
    side: int = 1


@dataclass
class XorEnd:
    split: El
    cont: Seq
    exit: Seq
    cont_label: str = ""
    exit_label: str = ""
    side: int = 1


Block = Union[Leaf, Seq, Split, Loop, XorEnd]


@dataclass
class Process:
    root: Seq
    els: dict[str, El]
    flows: list[Flow]
    domain: dict
    data: list[El] = field(default_factory=list)  # data objects / stores
    data_links: list[tuple[str, str]] = field(default_factory=list)  # (src, dst) dotted flows
    notes: list[El] = field(default_factory=list)
    note_links: list[tuple[str, str]] = field(default_factory=list)  # (element, note)
    # pools: main pool name ("" = no pool), lane names, partner pools
    pool_name: str = ""
    lanes: list[str] = field(default_factory=list)
    lanes_without_pool: bool = False
    partners: list[dict] = field(default_factory=list)  # {name, mode, side, els, flows, n_msg}


class ProcessGenerator:
    def __init__(self, rng: random.Random, style: str):
        self.rng = rng
        self.style = style
        self.dom = V.pick_domain(rng)
        self.els: dict[str, El] = {}
        self.next_id = 0
        self.count = {"loop": 0, "xorend": 0, "evgw": 0, "or": 0}
        r = rng.random()
        self.case = "lower" if r < 0.5 else "sentence" if r < 0.9 else "title" if r < 0.98 else "upper"
        r = rng.random()
        self.group_case = ("lower" if r < 0.2 else "sentence" if r < 0.55 else "title" if r < 0.88
                           else "upper")
        self.used_text: set[str] = set()

    # ---------------------------------------------------------------- elements
    def new(self, kind: str, **kw) -> El:
        self.next_id += 1
        el = El(id=f"x{self.next_id}", kind=kind, **kw)
        self.els[el.id] = el
        return el

    def n_flow(self) -> int:
        return sum(1 for e in self.els.values() if e.kind in FLOW_KINDS and e.pool == "main")

    def task(self) -> Leaf:
        return Leaf(self.new("task"))

    def inter(self, catch_only: bool = False) -> Leaf:
        r = self.rng.random()
        if catch_only:
            ev, throw = ("message", False) if r < 0.6 else ("timer", False)
        elif r < 0.44:
            ev, throw = "message", False
        elif r < 0.69:
            ev, throw = "timer", False
        elif r < 0.96:
            ev, throw = "message", True
        else:
            ev = self.rng.choice(["none", "signal", "conditional", "escalation"])
            throw = ev in ("none", "escalation") or self.rng.random() < 0.5
        return Leaf(self.new("inter", ev=ev, throw=throw))

    def end(self) -> Leaf:
        r = self.rng.random()
        ev = "none" if r < 0.78 else "message" if r < 0.94 else "terminate"
        return Leaf(self.new("end", ev=ev))

    def gateway(self, gw: str, split: bool) -> El:
        marker = ""
        if gw == "xor" and self.rng.random() < 0.15:
            marker = "blank"
        return self.new("gw", gw=gw, split=split, marker=marker)

    # ---------------------------------------------------------------- growth
    def generate(self, n_target: int) -> Seq:
        rng = self.rng
        r = rng.random()
        start = self.new("start", ev="none" if r < 0.45 else "message" if r < 0.9 else "timer")
        root = Seq([Leaf(start), self.task(), self.end()])
        ops = [("task", 4.0), ("event", 1.7), ("xor", 2.2), ("and", 1.3), ("loop", 0.7),
               ("xorend", 0.9), ("evgw", 0.35), ("or", 0.08)]
        guard = 0
        while self.n_flow() < n_target and guard < 400:
            guard += 1
            op = rng.choices([o for o, _ in ops], weights=[w for _, w in ops])[0]
            seqs = self._seqs(root)
            weights = [4.0 if d == 0 else 0.9 / d ** 1.5 for _, d in seqs]
            seq, depth = rng.choices(seqs, weights=weights)[0]
            lo, hi = _insert_range(seq)
            if lo > hi:
                continue
            pos = rng.randint(lo, hi)
            room = n_target - self.n_flow()
            if op == "task":
                seq.items.insert(pos, self.task())
            elif op == "event":
                seq.items.insert(pos, self.inter())
            elif op in ("xor", "and", "or", "evgw") and depth < 3 and room >= 3:
                if op in ("evgw", "or") and self.count[op] >= 1:
                    continue
                self._insert_split(seq, pos, op, room)
            elif op == "loop" and self.count["loop"] < 2 and room >= 2 and depth < 3:
                self._insert_loop(seq, lo, hi)
            elif op == "xorend" and self.count["xorend"] < 2 and room >= 3 and depth < 3:
                self._insert_xorend(seq, pos, lo, hi)
        used = {el.id for el in tree_elements(root)}
        for eid in [e for e in self.els if e not in used]:
            del self.els[eid]  # never drawn: must not be referenced by data objects / notes
        return root

    def _seqs(self, block: Block, depth: int = 0) -> list[tuple[Seq, int]]:
        out: list[tuple[Seq, int]] = []
        if isinstance(block, Seq):
            out.append((block, depth))
            for it in block.items:
                out += self._seqs(it, depth)
        elif isinstance(block, Split):
            for b in block.branches:
                out += self._seqs(b, depth + 1)
        elif isinstance(block, Loop):
            out += self._seqs(block.body, depth + 1)
        elif isinstance(block, XorEnd):
            out += self._seqs(block.cont, depth)
            out += self._seqs(block.exit, depth + 1)
        return out

    def _insert_split(self, seq: Seq, pos: int, op: str, room: int) -> None:
        rng = self.rng
        k = 2 if rng.random() < 0.72 or room < 5 else 3
        gw = "xor" if op in ("xor", "evgw") else op
        split = self.gateway("evgw" if op == "evgw" else gw, True)
        branches: list[Seq] = []
        avail = 0  # existing items that can become the first branch (a contiguous run)
        while avail < 2 and pos + avail < len(seq.items) and _wrappable(seq.items[pos + avail]):
            avail += 1
        take = 0
        if op != "evgw" and avail and rng.random() < 0.45:
            take = 1 if avail == 1 or rng.random() < 0.6 else 2
        first = Seq(seq.items[pos:pos + take]) if take else None
        if take:
            del seq.items[pos:pos + take]
        skip = op == "xor" and k == 2 and rng.random() < 0.18  # optional path: split -> merge
        for i in range(k):
            if i == 0 and first is not None:
                branches.append(first)
                continue
            if skip and i == 1:
                branches.append(Seq([]))
                continue
            if op == "evgw":
                extra = rng.choice([0, 1, 1])
                items: list = [self.inter(catch_only=True)] + [self.task() for _ in range(extra)]
            else:
                items = [self.task() for _ in range(rng.choice([1, 1, 1, 2]) if room > 6 else 1)]
            branches.append(Seq(items, lock_first=op == "evgw"))
        term = [False] * k
        if op in ("xor", "evgw") and k == 3 and rng.random() < 0.3:
            j = rng.randrange(1, 3)
            if branches[j].items:
                branches[j].items.append(self.end())
                term[j] = True
        if op in ("xor", "evgw") and rng.random() < 0.25:
            merge = self.new("task")  # implicit merge: the branches flow straight into a task
        else:
            merge = self.gateway("xor" if op == "evgw" else gw, False)
        labels = [""] * k
        if op in ("xor", "or"):
            r = rng.random()
            conds = V.conditions(rng, self.dom, k)
            if r < 0.8:
                labels = conds
            elif r < 0.9:
                labels[rng.randrange(k)] = conds[0]
        if op in ("evgw", "or"):
            self.count[op] += 1
        seq.items.insert(pos, Split(split, merge, branches, labels, term))

    def _insert_loop(self, seq: Seq, lo: int, hi: int) -> None:
        rng = self.rng
        cands = [i for i in range(lo, min(hi, len(seq.items))) if _wrappable(seq.items[i])]
        if not cands:
            return
        i = rng.choice(cands)
        j = i + 1
        while j < len(seq.items) and j - i < 3 and _wrappable(seq.items[j]) and rng.random() < 0.4:
            j += 1
        body = Seq(seq.items[i:j])
        direct = isinstance(body.items[0], Leaf) and body.items[0].el.kind == "task" and rng.random() < 0.2
        merge = None if direct else self.gateway("xor", False)
        split = self.gateway("xor", True)
        fwd = back = ""
        r = rng.random()
        if r < 0.55:
            fwd, back = rng.choice(V.LOOP_FWD), rng.choice(V.LOOP_BACK)
        elif r < 0.75:
            back = rng.choice(V.LOOP_BACK)
        side = 1 if rng.random() < 0.75 else -1
        seq.items[i:j] = [Loop(merge, split, body, fwd, back, side)]
        self.count["loop"] += 1

    def _insert_xorend(self, seq: Seq, pos: int, lo: int, hi: int) -> None:
        rng = self.rng
        if pos >= len(seq.items):
            pos = len(seq.items) - 1
        if pos < lo or not seq.items[pos:]:
            return
        if any(isinstance(it, XorEnd) for it in seq.items[pos:]):
            return
        cont = Seq(seq.items[pos:])
        n = rng.choice([0, 1, 1, 1, 2])
        exit_items = [self.task() for _ in range(n)]
        if rng.random() < 0.15:
            exit_items.insert(0, self.inter())
        exit_items.append(self.end())
        split = self.gateway("xor", True)
        a, b = V.conditions(rng, self.dom, 2)
        cl, el = (a, b) if rng.random() < 0.78 else ("", "") if rng.random() < 0.5 else ("", b)
        side = 1 if rng.random() < 0.6 else -1
        seq.items[pos:] = [XorEnd(split, cont, Seq(exit_items), cl, el, side)]
        self.count["xorend"] += 1

    # ---------------------------------------------------------------- labels
    def label(self, text: str) -> str:
        text = V.cased(text, self.case)
        if self.style == "sketch" and self.rng.random() < 0.03:
            text = V.typo(text, self.rng)
        return text

    def unique(self, make) -> str:
        for _ in range(12):
            t = make()
            if t not in self.used_text:
                break
        self.used_text.add(t)
        return t

    def decorate(self) -> None:
        rng, dom = self.rng, self.dom
        for el in self.els.values():
            if el.pool != "main":
                continue
            if el.kind == "task":
                el.text = self.label(self.unique(lambda: V.task_name(rng, dom)))
                r = rng.random()
                el.marker = ("sub" if r < 0.04 else "mi" if r < 0.06 else "mis" if r < 0.07
                             else "loop" if r < 0.085 else "")
                if self.style == "clean" and rng.random() < 0.12:
                    el.icon = rng.choice(["user", "service", "send", "receive", "script", "rule",
                                          "manual"])
            elif el.kind == "start":
                if rng.random() > 0.12:
                    kind = "timer" if el.ev == "timer" else "start"
                    el.text = self.label(self.unique(lambda k=kind: V.event_name(rng, dom, k)))
            elif el.kind == "end":
                if rng.random() > 0.14:
                    el.text = self.label(self.unique(lambda: V.event_name(rng, dom, "end")))
            elif el.kind == "inter":
                if rng.random() > (0.05 if el.ev == "timer" else 0.1):
                    if el.ev == "timer":
                        kind = "timer"
                    elif el.ev == "message":
                        kind = "throw" if el.throw else "catch"
                    else:
                        kind = "other"
                    if kind == "other":
                        el.text = self.label(rng.choice(["escalation", "signal sent", "condition met",
                                                         "deadline missed", "status changed",
                                                         "alarm", "milestone reached"]))
                    else:
                        el.text = self.label(self.unique(lambda k=kind: V.event_name(rng, dom, k)))
            elif el.kind == "gw":
                p = 0.12 if self.style == "clean" else 0.03
                if el.split and el.gw in ("xor", "or") and rng.random() < p:
                    q = rng.choice(V.GW_QUESTIONS).format(o=rng.choice(dom["objects"]))
                    el.text = self.label(q)

    def group_label(self, text: str) -> str:
        return V.cased(text, self.group_case)


def tree_elements(block: Block) -> list[El]:
    if isinstance(block, Leaf):
        return [block.el]
    if isinstance(block, Seq):
        return [e for it in block.items for e in tree_elements(it)]
    if isinstance(block, Split):
        out = [block.split] + ([block.merge] if block.merge else [])
        return out + [e for b in block.branches for e in tree_elements(b)]
    if isinstance(block, Loop):
        return [block.split] + ([block.merge] if block.merge else []) + tree_elements(block.body)
    if isinstance(block, XorEnd):
        return [block.split] + tree_elements(block.cont) + tree_elements(block.exit)
    raise TypeError(type(block))


def _insert_range(seq: Seq) -> tuple[int, int]:
    items = seq.items
    lo = 1 if seq.lock_first or (items and isinstance(items[0], Leaf) and items[0].el.kind == "start") else 0
    hi = len(items)
    if items and ((isinstance(items[-1], Leaf) and items[-1].el.kind == "end")
                  or isinstance(items[-1], XorEnd)):
        hi -= 1
    return lo, hi


def _wrappable(item: Block) -> bool:
    if isinstance(item, Leaf):
        return item.el.kind in ("task", "inter")
    return isinstance(item, (Split, Loop))


# --------------------------------------------------------------------------- #
# Flows
# --------------------------------------------------------------------------- #
def build_flows(root: Seq) -> list[Flow]:
    """Sequence flows of the tree, with branch labels and routing hints."""
    flows: list[Flow] = []

    def walk(b: Block) -> tuple[Optional[str], Optional[str], str]:
        """-> (entry id, exit id, pending label for the flow leaving the exit)."""
        if isinstance(b, Leaf):
            return b.el.id, b.el.id, ""
        if isinstance(b, Seq):
            entry = exit_ = None
            pending = ""
            for it in b.items:
                e_in, e_out, pend = walk(it)
                if exit_ is not None and e_in is not None:
                    flows.append(Flow(exit_, e_in, label=pending, hint="seq"))
                if entry is None:
                    entry = e_in
                if e_out is not None:
                    exit_, pending = e_out, pend
            return entry, exit_, pending
        if isinstance(b, Split):
            for br, lab, term in zip(b.branches, b.labels, b.term, strict=True):
                if not br.items:
                    flows.append(Flow(b.split.id, b.merge.id, label=lab, hint="skip"))
                    continue
                e_in, e_out, pend = walk(br)
                flows.append(Flow(b.split.id, e_in, label=lab, hint="split"))
                if not term:
                    flows.append(Flow(e_out, b.merge.id, label=pend, hint="join"))
            return b.split.id, b.merge.id, ""
        if isinstance(b, Loop):
            e_in, e_out, pend = walk(b.body)
            if b.merge is not None:
                flows.append(Flow(b.merge.id, e_in, hint="seq"))
                target = b.merge.id
            else:
                target = e_in
            flows.append(Flow(e_out, b.split.id, label=pend, hint="seq"))
            flows.append(Flow(b.split.id, target, label=b.back, hint="back", side=b.side))
            return (b.merge.id if b.merge is not None else e_in), b.split.id, b.fwd
        if isinstance(b, XorEnd):
            c_in, c_out, c_pend = walk(b.cont)
            x_in, _, _ = walk(b.exit)
            flows.append(Flow(b.split.id, c_in, label=b.cont_label, hint="split"))
            flows.append(Flow(b.split.id, x_in, label=b.exit_label, hint="split", side=b.side))
            return b.split.id, c_out, c_pend
        raise TypeError(type(b))

    walk(root)
    return flows


def generate_process(rng: random.Random, style: str, n_target: int) -> tuple[Process, ProcessGenerator]:
    gen = ProcessGenerator(rng, style)
    root = gen.generate(n_target)
    gen.decorate()
    flows = build_flows(root)
    proc = Process(root=root, els=gen.els, flows=flows, domain=gen.dom)
    _add_pools(proc, gen)
    _add_data(proc, gen)
    _add_notes(proc, gen)
    return proc, gen


def _add_pools(proc: Process, gen: ProcessGenerator) -> None:
    rng, dom = gen.rng, gen.dom
    r = rng.random()
    partner = rng.random() < 0.33
    if not partner and r < 0.3:
        return  # plain process, no pool
    if not partner and r < 0.33:
        proc.lanes_without_pool = True
        proc.lanes = [gen.group_label(x) for x in rng.sample(dom["roles"], 2)]
        return
    proc.pool_name = gen.group_label(rng.choice(dom["pools"]))
    r = rng.random()
    n_lanes = 1 if r < 0.42 else 2 if r < 0.8 else 3
    if n_lanes > 1:
        proc.lanes = [gen.group_label(x) for x in rng.sample(dom["roles"], n_lanes)]
    if partner:
        names = [n for n in dom["partners"]]
        rng.shuffle(names)
        n_partners = 1 if rng.random() < 0.82 else 2
        sides = [-1, 1] if n_partners == 2 else [-1 if rng.random() < 0.7 else 1]
        for i in range(n_partners):
            mode = "collapsed" if rng.random() < 0.72 or i == 1 else "expanded"
            p = {"name": gen.group_label(names[i % len(names)]), "mode": mode, "side": sides[i],
                 "n_msg": rng.choice([1, 2, 2, 3, 3, 4]), "els": [], "multi": rng.random() < 0.08}
            if mode == "expanded":
                p["els"] = _partner_process(gen, f"partner{i}")
            proc.partners.append(p)


def _partner_process(gen: ProcessGenerator, key: str) -> list[El]:
    """A short sequence (start, 1-3 tasks / events, end) for an expanded partner pool."""
    rng, dom = gen.rng, gen.dom
    els = [gen.new("start", ev="none" if rng.random() < 0.5 else "message", pool=key)]
    for _ in range(rng.choice([1, 2, 2, 3])):
        if rng.random() < 0.3:
            el = gen.new("inter", ev="message", throw=rng.random() < 0.5, pool=key)
            el.text = gen.label(V.event_name(rng, dom, "throw" if el.throw else "catch"))
        else:
            el = gen.new("task", pool=key)
            el.text = gen.label(gen.unique(lambda: V.task_name(rng, dom)))
        els.append(el)
    els.append(gen.new("end", ev="none" if rng.random() < 0.7 else "message", pool=key))
    for el in (els[0], els[-1]):
        if rng.random() < 0.85:
            el.text = gen.label(V.event_name(rng, dom, "start" if el.kind == "start" else "end"))
    return els


def _add_data(proc: Process, gen: ProcessGenerator) -> None:
    rng, dom = gen.rng, gen.dom
    n = rng.choices([0, 1, 2, 3], weights=[22, 30, 27, 21])[0]
    tasks = [e for e in proc.els.values() if e.kind == "task" and e.pool == "main"]
    if not tasks:
        return
    succ: dict[str, list[str]] = {}
    for f in proc.flows:
        succ.setdefault(f.src, []).append(f.dst)
    for _ in range(n):
        store = rng.random() < 0.28
        d = gen.new("store" if store else "data")
        d.text = gen.label(rng.choice(dom["stores"]) if store else V.data_name(rng, dom))
        if rng.random() < 0.06:
            d.text = ""
        t = rng.choice(tasks)
        r = rng.random()
        links: list[tuple[str, str]] = []
        if r < 0.4:
            links.append((t.id, d.id))
        elif r < 0.75:
            links.append((d.id, t.id))
        else:
            links.append((t.id, d.id))
            nxt = [s for s in succ.get(t.id, []) if proc.els[s].kind == "task"]
            if nxt:
                links.append((d.id, nxt[0]))
        proc.data.append(d)
        proc.data_links += links


def _add_notes(proc: Process, gen: ProcessGenerator) -> None:
    rng = gen.rng
    n = rng.choices([0, 1, 2], weights=[70, 22, 8])[0]
    cands = [e for e in proc.els.values() if e.pool == "main" and e.kind in ("task", "gw", "inter",
                                                                               "start", "end")]
    for _ in range(n):
        target = rng.choice([c for c in cands if c.kind == "task"] or cands)
        note = gen.new("note")
        note.text = gen.label(rng.choice(V.NOTES)) if rng.random() < 0.85 else V.NOTES[0]
        proc.notes.append(note)
        proc.note_links.append((target.id, note.id))
