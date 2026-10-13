"""Synthetic PlantUML activity diagrams (new syntax): start / stop / end, actions (some multi-line),
if / elseif / else with guards, switch, while, repeat, fork, notes, partitions and - in about half
of them - swimlanes with flows crossing lanes (the BPMN-like case). SDL action endings (`:x>`,
`:x<`) are not used: PlantUML 1.2026.8 rejects them before stop / end / fork / if / while and
after else.

The generator builds a statement tree, writes it as PlantUML and, walking the same tree, derives
the graph PlantUML draws for it (the Diagram the importer should produce, see puml_check): a merge
node where two or more flows rejoin after endif / endswitch, while loops back into their decision,
repeat entered through a merge node (or its `repeat :label;` box), fork / join bars, nodes owned
by the lane / partition they are created in, notes anchored to the action before them."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from diagram_ir import Diagram, Group

from .puml_check import Intent
from .puml_words import Domain

LABEL_PAIRS = (("yes", "no"), ("yes", "no"), ("true", "false"), ("valid", "invalid"),
               ("approved", "rejected"), ("ok", "error"), ("found", "not found"), ("Y", "N"),
               ("pass", "fail"), ("accepted", "declined"))
DETAILS = ("(within 24h)", "via email", "for each item", "in the ERP", "and log it", "manually", "if needed",
           "with retries", "by phone", "in batch")
NOTE_TEXTS = ("SLA: 2 business days", "automated step", "manual approval", "may take a while",
              "see policy 4.2",
              "runs nightly", "audit logged", "customer notified by SMS", "requires two signatures",
              "uses external API", "idempotent", "retried 3 times", "done by a supervisor", "optional step",
              "only for VIP customers", "timeout after 30 min")
LOOP_EXIT = ("no", "done", "false", "finished", "")
CASES = ("A", "B", "C", "Standard", "Express", "Priority", "Card", "Cash", "Invoice", "Other", "Low", "High",
         "Domestic", "International")
PARTITIONS = ("Preparation", "Validation", "Processing", "Fulfilment", "Billing", "Review", "Closing",
              "Intake",
              "Execution", "Follow-up")


@dataclass
class _S:
    """One statement. kind: action | if | switch | while | repeat | fork | partition | lane | note | arrow
    | stop | end | kill | detach."""

    kind: str
    text: str = ""
    shape: str = ";"  # action ending: ; > <
    lines: list[str] = field(default_factory=list)  # action text as written over several lines
    # (condition, label, body, prefix label of the edge from the previous decision)
    branches: list[tuple[str, str, list[_S], str]] = field(default_factory=list)
    else_body: Optional[list[_S]] = None
    else_label: str = ""
    body: list[_S] = field(default_factory=list)
    labels: tuple[str, str] = ("", "")  # loops: (stay, exit)
    start_text: str = ""  # repeat :label;
    side: str = "right"
    floating: bool = False
    end_kind: str = "fork"  # fork: end fork | end merge


class _Gen:
    def __init__(self, rng: random.Random, dom: Domain, colors: bool):
        self.rng = rng
        self.dom = dom
        self.colors = colors
        self.tasks = list(dom.tasks)
        rng.shuffle(self.tasks)
        self.combos = [f"{t} {d}" for t in self.tasks for d in DETAILS]  # when the plain tasks run out
        rng.shuffle(self.combos)
        self.checks = list(dom.checks)
        rng.shuffle(self.checks)
        self.notes = list(NOTE_TEXTS)
        rng.shuffle(self.notes)
        self.lanes: list[str] = []
        self.cur_lane = ""
        self.lane_rate = 0.0
        self.pair = rng.choice(LABEL_PAIRS)
        self.labelled = rng.random() < 0.7
        self.n_tasks = 0

    # ---- tree generation ---------------------------------------------------------------
    def task(self) -> _S:
        rng = self.rng
        k = self.n_tasks - len(self.tasks)
        if k < 0:
            text = self.tasks[self.n_tasks]
        elif k < len(self.combos):
            text = self.combos[k]
        else:
            text = f"{rng.choice(self.tasks)} ({k + 2})"
        self.n_tasks += 1
        s = _S("action", text)
        r = rng.random()
        if r < 0.1:
            s.lines = [text, rng.choice(DETAILS)]
        elif r < 0.16 and " " in text:
            head, tail = text.rsplit(" ", 1)
            s.lines = [head, tail]
        s.text = "\n".join(s.lines) if s.lines else text
        return s

    def check(self) -> str:
        return self.checks.pop() if self.checks else self.rng.choice(self.dom.checks)

    def labels(self) -> tuple[str, str]:
        if not self.labelled:
            return "", ""
        return self.pair if self.rng.random() < 0.85 else self.rng.choice(LABEL_PAIRS)

    def maybe_lane(self, out: list[_S], allow: bool) -> None:
        if out and out[-1].kind == "lane":  # never two switches in a row (an empty lane column)
            return
        if allow and len(self.lanes) > 1 and self.rng.random() < self.lane_rate:
            other = self.rng.choice([x for x in self.lanes if x != self.cur_lane])
            self.cur_lane = other
            out.append(_S("lane", other))

    def block(self, budget: int, depth: int, lanes_ok: bool = True) -> list[_S]:
        """Statements holding about `budget` actions (at least one)."""
        rng = self.rng
        out: list[_S] = []
        while budget > 0:
            self.maybe_lane(out, lanes_ok)
            r = rng.random()
            used = 1
            if depth < 3 and budget >= 3 and r < 0.2:
                st, used = self.if_stmt(budget, depth, lanes_ok)
            elif depth < 3 and budget >= 2 and r < 0.27:
                st, used = self.loop(budget, depth, lanes_ok, "while")
            elif depth < 3 and budget >= 2 and r < 0.32:
                st, used = self.loop(budget, depth, lanes_ok, "repeat")
            elif depth < 3 and budget >= 3 and r < 0.39:
                st, used = self.fork(budget, depth, lanes_ok)
            elif depth < 2 and budget >= 3 and r < 0.42:
                st, used = self.switch(budget, depth, lanes_ok)
            elif depth < 2 and budget >= 2 and r < 0.47 and (not self.lanes or rng.random() < 0.3):
                st = _S("partition", rng.choice(PARTITIONS))
                used = rng.randint(1, min(budget, 4))
                st.body = self.block(used, depth + 1, lanes_ok=False)
            else:
                st = self.task()
                if out and out[-1].kind == "action" and rng.random() < 0.07:  # plain flow, no guard
                    out.append(_S("arrow", rng.choice(["ok", "done", "next", "approved", "on success",
                                                       "data ready", "confirmed"])))
            out.append(st)
            if st.kind == "action" and rng.random() < 0.07 and self.notes:
                out.append(_S("note", self.notes.pop(), side=rng.choice(["left", "right"]),
                              floating=rng.random() < 0.1))
            budget -= used
        return out

    def split(self, budget: int, parts: int) -> list[int]:
        if budget > parts:
            cuts = sorted(self.rng.sample(range(1, budget), parts - 1))
        else:
            cuts = list(range(1, parts))
        sizes = [b - a for a, b in zip([0] + cuts, cuts + [budget], strict=True)]
        return [max(1, x) for x in sizes]

    def terminate(self, body: list[_S]) -> None:
        at = len(body) - 1 if body and body[-1].kind == "lane" else len(body)  # before the lane return
        kind = self.rng.choices(["stop", "end", "detach", "kill"], weights=(45, 40, 10, 5))[0]
        if kind in ("detach", "kill") and not (at and body[at - 1].kind in ("action", "note")):
            kind = "stop"  # after a loop / decision the open exit would be drawn going nowhere
        body.insert(at, _S(kind))

    def body(self, budget: int, depth: int, lanes_ok: bool, lead: float = 0.0) -> list[_S]:
        """A branch / loop body that ends in the lane it started in. PlantUML draws the node that
        closes a block (merge, join, loop decision) in the block's own lane while the importer
        uses the lane current at the closing keyword; returning to the lane keeps both equal."""
        lane0 = self.cur_lane
        out: list[_S] = []
        if lead and self.rng.random() < lead:
            self.maybe_lane(out, lanes_ok)
        out += self.block(budget, depth + 1, lanes_ok)
        if self.cur_lane != lane0:
            out.append(_S("lane", lane0))
            self.cur_lane = lane0
        return out

    def if_stmt(self, budget: int, depth: int, lanes_ok: bool) -> tuple[_S, int]:
        rng = self.rng
        used = rng.randint(2, min(budget, 7))
        n_branch = rng.choices([1, 2, 3], weights=(70, 22, 8))[0]
        has_else = rng.random() < 0.75
        k = min(used, n_branch + has_else)
        n_branch = max(1, k - has_else)
        parts = n_branch + has_else
        sizes = self.split(used, parts) if used >= parts else [1] * parts
        st = _S("if")
        yes, no = self.labels()
        for i in range(n_branch):
            body = self.body(sizes[i], depth, lanes_ok, lead=0.25)
            prefix = no if (i > 0 and yes and rng.random() < 0.5) else ""
            st.branches.append((self.check(), yes if i == 0 or rng.random() < 0.8 else "", body, prefix))
        if has_else:
            st.else_body, st.else_label = self.body(sizes[-1], depth, lanes_ok, lead=0.3), no
        live = len(st.branches) + 1  # the else branch or the implicit false path
        if live >= 2 and rng.random() < 0.18:  # one branch ends the flow
            bodies = [b for _c, _l, b, _p in st.branches]
            if st.else_body is not None:
                bodies.append(st.else_body)
            self.terminate(rng.choice(bodies))
        return st, used

    def switch(self, budget: int, depth: int, lanes_ok: bool) -> tuple[_S, int]:
        """No lane changes inside cases: PlantUML 1.2026.8 then draws some case boxes twice."""
        rng = self.rng
        used = rng.randint(3, min(budget, 7))
        n = rng.randint(2, min(4, used))
        st = _S("switch", self.check())
        cases = rng.sample(CASES, n)
        for label, size in zip(cases, self.split(used, n), strict=True):
            st.branches.append(("", label, self.body(size, depth, False), ""))
        return st, used

    def loop(self, budget: int, depth: int, lanes_ok: bool, kind: str) -> tuple[_S, int]:
        rng = self.rng
        used = rng.randint(1, min(budget, 5))
        st = _S(kind, self.check())
        stay, leave = self.labels()
        if stay and rng.random() < 0.3:
            leave = rng.choice(LOOP_EXIT)
        st.labels = (stay, leave)
        st.body = self.body(used, depth, lanes_ok)
        # `repeat :label;` only on the default look: several themes draw that box's text invisibly
        if kind == "repeat" and self.colors and rng.random() < 0.2:
            st.start_text = self.task().text.split("\n")[0]
        return st, used

    def fork(self, budget: int, depth: int, lanes_ok: bool) -> tuple[_S, int]:
        """`end merge` only without swimlanes: across lanes PlantUML 1.2026.8 leaves branches dangling."""
        rng = self.rng
        used = rng.randint(2, min(budget, 7))
        n = rng.randint(2, min(4, used))
        st = _S("fork")
        st.end_kind = "merge" if (not self.lanes and rng.random() < 0.08) else "fork"
        for size in self.split(used, n):
            st.branches.append(("", "", self.body(size, depth, lanes_ok, lead=0.2), ""))
        return st, used


class _Walk:
    """Write the statement tree as PlantUML and derive the drawn graph from it."""

    def __init__(self, rng: random.Random, colors: bool, lane_colors: dict[str, str]):
        self.rng = rng
        self.it = Intent("activity", "TB")
        self.out: list[str] = []
        self.lane: Optional[Group] = None
        self.lanes: dict[str, Group] = {}
        self.lane_colors = lane_colors
        self.regions: list[Group] = []
        self.arrow_label = ""
        self.last: Optional[str] = None
        self.indent = ""
        self.colors = colors

    def node(self, kind: str, text: str = "") -> str:
        nid = self.it.node(kind, text)
        if self.lane is not None:
            self.lane.members.append(nid)
        if self.regions:
            self.regions[-1].members.append(nid)
        return nid

    def enter(self, tails: list[tuple[str, str]], nid: str) -> list[tuple[str, str]]:
        for t, lbl in tails:
            self.it.edge(t, nid, "flow", lbl or self.arrow_label)
        self.arrow_label = ""
        self.last = nid
        return [(nid, "")]

    def join(self, ends: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if len(ends) < 2:
            return ends
        m = self.node("merge")
        for t, lbl in ends:
            self.it.edge(t, m, "flow", lbl)
        self.last = m
        return [(m, "")]

    def emit(self, line: str) -> None:
        self.out.append(self.indent + line)

    def walk(self, stmts: list[_S], tails: list[tuple[str, str]]) -> list[tuple[str, str]]:
        rng = self.rng
        for st in stmts:
            k = st.kind
            if k == "lane":
                color = self.lane_colors.get(st.text, "")
                self.out.append(f"|{color}|{st.text}|" if color else f"|{st.text}|")
                if st.text not in self.lanes:
                    self.lanes[st.text] = self.it.group("lane", st.text)
                self.lane = self.lanes[st.text]
            elif k == "action":
                if st.lines:  # escaped breaks only: a source line like `if needed;` is misread
                    self.emit(":" + "\\n".join(st.lines) + st.shape)
                else:
                    self.emit(f":{st.text}{st.shape}")
                kind = {">": "send_signal", "<": "receive_signal"}.get(st.shape, "action")
                tails = self.enter(tails, self.node(kind, st.text))
            elif k == "arrow":
                self.emit(f"-> {st.text};")
                self.arrow_label = st.text
            elif k == "note":
                head = ("floating note" if st.floating else "note") + f" {st.side}"
                parts = st.text.split(" ", 2)
                if len(parts) == 3 and rng.random() < 0.3:
                    lines = [" ".join(parts[:2]), parts[2]]
                    text = "\n".join(lines)
                    self.emit(head)
                    self.out += [self.indent + "  " + x for x in lines]
                    self.emit("end note")
                else:
                    text = st.text
                    self.emit(f"{head}{rng.choice([': ', ' : '])}{text}")
                nid = self.node("note", text)
                if not st.floating and self.last is not None:
                    self.it.edge(nid, self.last, "anchor")
            elif k in ("stop", "end"):
                self.emit(k)
                tails = self.enter(tails, self.node("final" if k == "stop" else "flow_final"))
                tails = []
            elif k in ("kill", "detach"):
                self.emit(k)
                tails = []
            elif k == "if":
                tails = self.if_block(st, tails)
            elif k == "switch":
                self.emit(f"switch ({st.text})")
                d = self.node("decision", st.text)
                tails = self.enter(tails, d)
                ends: list[tuple[str, str]] = []
                for _c, label, body, _p in st.branches:
                    self.emit(f"case ({label})")
                    ends += self.nested(body, [(d, label)])
                self.emit("endswitch")
                tails = self.join(ends)
            elif k == "while":
                stay, leave = st.labels
                self.emit(f"while ({st.text})" + (f" is ({stay})" if stay else ""))
                d = self.node("decision", st.text)
                tails = self.enter(tails, d)
                for t, lbl in self.nested(st.body, [(d, stay)]):
                    self.it.edge(t, d, "flow", lbl or self.arrow_label)
                self.arrow_label = ""
                self.emit("endwhile" + (f" ({leave})" if leave else ""))
                self.last = d
                tails = [(d, leave)]
            elif k == "repeat":
                stay, leave = st.labels
                if st.start_text:
                    self.emit(f"repeat :{st.start_text};")
                    top = self.node("action", st.start_text)
                else:
                    self.emit("repeat")
                    top = self.node("merge")
                tails = self.enter(tails, top)
                tails = self.nested(st.body, tails)
                head = f"repeat while ({st.text})"
                if stay:
                    head += f" is ({stay})"
                if leave:
                    head += f" not ({leave})"
                self.emit(head)
                d = self.node("decision", st.text)
                self.enter(tails, d)
                self.it.edge(d, top, "flow", stay)
                tails = [(d, leave)]
            elif k == "fork":
                self.emit("fork")
                f = self.node("fork")
                tails = self.enter(tails, f)
                ends = []
                for i, (_c, _l, body, _p) in enumerate(st.branches):
                    if i:
                        self.emit("fork again")
                    ends += self.nested(body, [(f, "")])
                self.emit("end merge" if st.end_kind == "merge" else rng.choice(["end fork", "end fork",
                                                                                 "fork end"]))
                if ends:
                    j = self.node("merge" if st.end_kind == "merge" else "join")
                    for t, lbl in ends:
                        self.it.edge(t, j, "flow", lbl)
                    self.last = j
                    tails = [(j, "")]
                else:
                    tails = []
            elif k == "partition":
                quoted = " " in st.text or rng.random() < 0.3
                name = f'"{st.text}"' if quoted else st.text
                # `group X` ... `end group` draws a "use a bracket" warning into the image
                kw = "group" if rng.random() < 0.08 else "partition"
                self.emit(f"{kw} {name} {{")
                parent = self.regions[-1].id if self.regions else None
                self.regions.append(self.it.group("region", st.text, parent))
                tails = self.nested(st.body, tails)
                self.regions.pop()
                self.emit("}")
        return tails

    def nested(self, body: list[_S], tails: list[tuple[str, str]]) -> list[tuple[str, str]]:
        old = self.indent
        self.indent += "  " if self.rng.random() < 0.85 else "\t"
        tails = self.walk(body, tails)
        self.indent = old
        return tails

    def if_block(self, st: _S, tails: list[tuple[str, str]]) -> list[tuple[str, str]]:
        ends: list[tuple[str, str]] = []
        prev = ""
        for i, (cond, label, body, prefix) in enumerate(st.branches):
            then = f" then ({label})" if label else (" then" if self.rng.random() < 0.9 else "")
            if i == 0:
                self.emit(f"if ({cond}){then}")
                d = self.node("decision", cond)
                tails = self.enter(tails, d)
            else:
                self.emit((f"({prefix}) " if prefix else "") + f"elseif ({cond}){then}")
                d = self.node("decision", cond)
                self.it.edge(prev, d, "flow", prefix)
                self.last = d
            ends += self.nested(body, [(d, label)])
            prev = d
        if st.else_body is not None:
            self.emit(f"else ({st.else_label})" if st.else_label else "else")
            ends += self.nested(st.else_body, [(prev, st.else_label)])
        else:
            ends.append((prev, ""))  # the false path leaves the last decision
        self.emit(self.rng.choice(["endif", "endif", "end if"]))
        return self.join(ends)


def _lanes_used(stmts: list[_S]) -> set[str]:
    out = set()
    for st in stmts:
        if st.kind == "lane":
            out.add(st.text)
        for body in [b for _c, _l, b, _p in st.branches] + [st.body, st.else_body or []]:
            out |= _lanes_used(body)
    return out


def generate(rng: random.Random, dom: Domain, direction: Optional[str] = "TB",
             colors: bool = True, lanes: Optional[bool] = None) -> tuple[list[str], Diagram]:
    g = _Gen(rng, dom, colors)
    if lanes is None:
        lanes = rng.random() < 0.5
    lane_colors: dict[str, str] = {}
    if lanes:
        pool = list(dom.lanes)
        rng.shuffle(pool)
        g.lanes = pool[:rng.choices([2, 3, 4, 5], weights=(30, 38, 22, 10))[0]]
        g.cur_lane = g.lanes[0]
        g.lane_rate = rng.uniform(0.25, 0.5)
        if colors and rng.random() < 0.2:
            for name in g.lanes:
                if rng.random() < 0.6:
                    lane_colors[name] = rng.choice(["#AntiqueWhite", "#LightBlue", "#E8F5E9", "#FFF3E0",
                                                    "#F3E5F5", "#ECEFF1"])
    n_actions = rng.choices(range(3, 31), weights=[4, 6, 8, 9, 10, 10, 9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2,
                                                   2, 2, 1, 1, 1, 1, 1, 1, 1])[0]
    body = g.block(n_actions, 0)
    has_start = rng.random() < 0.92
    if lanes and not _lanes_used(body) - {g.lanes[0]}:
        # PlantUML draws no lane header for a single lane (the target would still have it): move
        # the flow to a second lane once. The body has no lane statements, so no block is affected.
        spots = [k for k in range(1, len(body))
                 if body[k].kind not in ("note", "arrow", "lane")
                 and body[k - 1].kind not in ("lane", "arrow")]
        if spots or has_start:
            body.insert(rng.choice(spots) if spots else 0, _S("lane", g.lanes[1]))
        else:
            lanes = False
    w = _Walk(rng, colors, lane_colors)
    tails: list[tuple[str, str]] = []
    if lanes:
        body.insert(0, _S("lane", g.lanes[0]))
    pre = [body.pop(0)] if lanes else []
    w.walk(pre, tails)
    if has_start:
        w.emit("start")
        tails = w.enter(tails, w.node("initial"))
    tails = w.walk(body, tails)
    kinds = {n.id: n.kind for n in w.it.d.nodes}
    # A loop exit or decision path with nothing after it is still drawn (with its label) but has no
    # edge in the target: such a flow always ends in a final node.
    dangling = any(lbl or kinds.get(t) == "decision" for t, lbl in tails)
    if tails:
        r = rng.random()
        if r < 0.7 or (dangling and r < 0.85):
            w.emit("stop")
            w.enter(tails, w.node("final"))
        elif r < 0.85 or dangling:
            w.emit("end")
            w.enter(tails, w.node("flow_final"))
    return w.out, w.it.d
