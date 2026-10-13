"""Synthetic PlantUML sequence diagrams: 2-8 participants of every kind (aliases, stereotypes, boxes),
3-25 messages built from nested calls (sync / async / reply / create / destroy, self calls, the
three activation spellings including `return`), notes in all positions, and alt / else / opt /
loop / par / break / critical / group / ref fragments. `generate` returns the PlantUML body and
the Diagram the importer should produce for it (see puml_check)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from diagram_ir import Diagram, Group

from .puml_check import Intent
from .puml_words import Domain, camel, ident, is_keyword, words

BOX_TITLES = ("Backend", "Internal Services", "Data Layer", "Cloud", "Core Platform", "Frontend",
              "Partner Side",
              "Payment Domain", "Our System", "Infrastructure")
BOX_COLORS = ("", "", "#LightBlue", "#EEEEFF", "#FFF7E0", "#E8F5E9", "#F3E5F5", "#LightYellow")
STEREOS = ("Service", "UI", "API", "Gateway", "Microservice", "External", "Legacy", "Worker")
REPLIES = ("OK", "200 OK", "ack", "true", "false", "token", "result", "confirmation", "error", "done", "id",
           "list of items", "status", "201 Created", "404 Not Found", "accepted", "rejected", "details",
           "data", "receipt")
NOTE_TEXTS = ("validates JWT", "retries 3 times", "fire and forget", "cached for 5 min", "idempotent call",
              "requires 2FA", "timeout 30s", "logged for audit", "user sees a spinner", "runs in background",
              "rate limited", "uses TLS", "may take a while", "batched nightly", "compensating action",
              "read replica", "eventually consistent", "manual step", "async callback", "sent by email")
LOOP_GUARDS = ("for each item", "3 times", "until confirmed", "every 5 min", "while retries remain",
               "for each order line", "polling", "for all recipients", "until queue empty", "10 times")
GROUP_LABELS = ("Checkout", "Retry logic", "Authentication", "Payment flow", "Validation", "Cleanup", "Setup",
                "Notification", "Error handling", "Sync")
CRITICAL_GUARDS = ("lock acquired", "DB transaction", "single writer", "atomic update")
ELSE_GUARDS = ("otherwise", "invalid", "declined", "timeout", "error", "not found", "")


@dataclass(eq=False)  # identity semantics: participants are compared by object
class _P:
    code: str
    text: str
    kind: str
    nid: str = ""
    stereo: Optional[str] = None
    declared: bool = True
    created: bool = False  # declared with `create` in the middle of the flow
    dead: bool = False  # destroyed


class _Seq:
    def __init__(self, rng: random.Random, dom: Domain, colors: bool):
        self.rng = rng
        self.dom = dom
        self.colors = colors  # explicit colours only on the default look (themes clash with them)
        self.it = Intent("sequence")
        self.lines: list[str] = []
        self.parts: list[_P] = []
        self.order: list[_P] = []  # creation order, as the importer sees it
        self.holders: list[Group] = []  # innermost fragment operand / group last
        self.budget = 0
        self.act = rng.choices(["none", "explicit", "shorthand", "return"], weights=(40, 28, 20, 12))[0]
        self.note_texts = list(NOTE_TEXTS) + [t.lower() for t in dom.tasks]
        rng.shuffle(self.note_texts)
        self.color_arrows = colors and rng.random() < 0.06
        self.last_msg: Optional[tuple[_P, _P]] = None
        self.box: Optional[Group] = None
        self.gap: Optional[str] = None  # `...` / `|||` waiting for the next message line

    # ---- participants ---------------------------------------------------------------
    def cast(self) -> None:
        rng, dom = self.rng, self.dom
        n = rng.choices(range(2, 9), weights=(10, 20, 22, 18, 13, 9, 8))[0]
        taken: set[str] = set()
        roles: list[tuple[str, str]] = []  # (kind, text)

        def take(pool, k: int) -> list[str]:
            cands = [x for x in pool if x.lower() not in taken]
            rng.shuffle(cands)
            out = cands[:k]
            taken.update(x.lower() for x in out)
            return out

        if rng.random() < 0.7:
            roles += [("actor", t) for t in take(dom.actors, 1)]
        if n >= 4 and rng.random() < 0.25:
            roles += [(rng.choice(["boundary", "participant"]), t)
                      for t in take(["Web UI", "Mobile App", "Kiosk", "Admin Console", "Browser"], 1)]
        n_store = 0 if n < 3 else rng.choices([0, 1, 2], weights=(35, 50, 15))[0]
        n_svc = max(1, n - len(roles) - n_store)
        for t in take(dom.services, n_svc):
            kind = rng.choices(["participant", "control", "entity", "boundary"], weights=(70, 14, 8, 8))[0]
            roles.append((kind, t))
        for t in take(dom.stores, n_store):
            kind = rng.choices(["database", "queue", "collections", "entity"], weights=(60, 20, 12, 8))[0]
            if any(w in t.lower() for w in ("queue", "events", "stream", "bus")):
                kind = "queue"
            roles.append((kind, t))
        if len(roles) < 2:
            roles += [("participant", t) for t in take(dom.services, 2 - len(roles))]
        aliases: set[str] = set()
        for kind, text in roles:
            p = _P(code="", text=text, kind=kind)
            simple = text.isidentifier() and not is_keyword(text)
            r = rng.random()
            if simple and r < 0.55:
                p.code = text
            else:
                base = ident(text) if r < 0.75 or not simple else "".join(w[0] for w in words(text)).upper()
                if len(base) < 2 or base.lower() in aliases or is_keyword(base):
                    base = ident(text)
                p.code = base
            k = 2
            while p.code.lower() in aliases:  # e.g. actor "Fleet Manager" vs participant FleetManager
                p.code, k = f"{ident(text)}{k}", k + 1
            aliases.add(p.code.lower())
            if kind == "participant" and rng.random() < 0.06:
                p.stereo = rng.choice(STEREOS)
            self.parts.append(p)

    def declare_line(self, p: _P) -> str:
        rng = self.rng
        kw = p.kind
        if p.code == p.text:
            head = f"{kw} {p.code}"
        elif not p.text.isidentifier() or rng.random() < 0.7:
            head = f'{kw} "{p.text}" as {p.code}'
        elif rng.random() < 0.5:
            head = f"{kw} {p.text} as {p.code}"
        else:
            head = f'{kw} {p.code} as "{p.text}"'
        if p.stereo:
            head += f" <<{p.stereo}>>"
        return head

    def declare_all(self) -> None:
        rng = self.rng
        plain = [p for p in self.parts if p.kind == "participant" and p.code == p.text and not p.stereo]
        for p in plain:
            if rng.random() < 0.15:
                p.declared = False  # appears on first use
        box: Optional[tuple[int, int]] = None
        decl = [p for p in self.parts if p.declared]
        if len(decl) >= 3 and rng.random() < 0.12:
            k = rng.randint(2, min(3, len(decl) - 1))
            start = rng.randint(1, len(decl) - k)
            box = (start, start + k)
        for i, p in enumerate(decl):
            if box and i == box[0]:
                title = rng.choice(BOX_TITLES)
                color = rng.choice(BOX_COLORS) if self.colors else ""
                self.lines.append(f'box "{title}" {color}'.rstrip())
                self.box = self.it.group("box", title)
            p.nid = self.it.node(p.kind, p.text, p.stereo)
            self.order.append(p)
            self.lines.append(self.declare_line(p))
            if box and box[0] <= i < box[1] and self.box is not None:
                self.box.members.append(p.nid)
            if box and i == box[1] - 1:
                self.lines.append("end box")

    def touch(self, p: _P) -> None:
        if not p.nid:  # implicit participant: created on first mention
            p.nid = self.it.node(p.kind, p.text, p.stereo)
            self.order.append(p)

    # ---- messages ---------------------------------------------------------------
    def label(self, kind: str, src: _P, tgt: _P) -> str:
        rng, dom = self.rng, self.dom
        if rng.random() < 0.03:
            return ""
        if kind == "reply":
            if rng.random() < 0.5:
                return rng.choice(REPLIES)
            return camel(words(rng.choice(dom.fields))) if rng.random() < 0.6 else \
                camel(words(rng.choice(dom.entities))) + rng.choice(["", "Id", "List", "Dto"])
        if kind == "async":
            ent = rng.choice(dom.entities)
            r = rng.random()
            if r < 0.5:
                return ent + rng.choice(["Created", "Updated", "Placed", "Cancelled", "Changed", "Requested"])
            if r < 0.8:
                return f"publish({camel(words(ent))}Event)"
            return rng.choice(["notify", "enqueue", "emit", "send"]) + f"({camel(words(ent))})"
        if kind == "self":
            return rng.choice(["validate()", "log event", "hash(password)", "check limits", "build response",
                               f"calculate{ident(rng.choice(dom.fields))}()",
                               f"{camel(words(rng.choice(dom.verbs)))}()", "update cache", "map to DTO",
                               "retry"])
        r = rng.random()
        verb = rng.choice(dom.verbs)
        ent = rng.choice(dom.entities)
        if r < 0.55:
            args = rng.sample(dom.fields, rng.choice([0, 1, 1, 2]))
            return f"{camel(words(verb) + words(ent)[-1:])}({', '.join(camel(words(a)) for a in args)})"
        if r < 0.8:
            return f"{verb} {' '.join(words(ent))}"
        method = rng.choice(["GET", "POST", "PUT", "DELETE", "PATCH"])
        return f"{method} /{'-'.join(words(ent))}s" + (f"/{rng.randint(1, 999)}" if method != "POST" else "")

    def msg(self, src: _P, tgt: _P, kind: str, label: str, mods: str = "") -> None:
        rng = self.rng
        if self.gap:  # PlantUML rejects `...` before create / deactivate / ...: only before a message
            self.lines.append(self.gap)
            self.gap = None
        reverse = src is not tgt and not mods and rng.random() < 0.06
        for p in ((tgt, src) if reverse else (src, tgt)):  # the importer meets the left name first
            self.touch(p)
        color = f"[#{rng.choice(['red', 'blue', 'green', 'gray', 'orange'])}]" if self.color_arrows and \
            rng.random() < 0.3 else ""
        if kind in ("sync", "create", "destroy"):
            if reverse:
                arrow = f"<{color}-"
            else:
                arrow = rng.choices([f"-{color}>", f"-{color}>o"], weights=(97, 3))[0]
        elif kind == "async":
            arrow = f"-{color}>>" if not reverse else f"<<{color}-"
        else:  # reply
            arrow = rng.choices([f"-{color}->", f"-{color}->>"], weights=(88, 12))[0] if not reverse \
                else f"<-{color}-"
        if reverse:
            line = f"{tgt.code} {arrow} {src.code}"
        else:
            line = f"{src.code} {arrow} {tgt.code}"
        if mods:
            line += f" {mods}"
        if label:
            line += f" : {label}"
        self.lines.append(line)
        self.emit_edge(src, tgt, kind, label)

    def emit_edge(self, src: _P, tgt: _P, kind: str, label: str) -> None:
        eid = self.it.edge(src.nid, tgt.nid, kind, label)
        if self.holders:
            self.holders[-1].members.append(eid)
        self.last_msg = (src, tgt)
        self.budget -= 1

    def note_after_message(self) -> None:
        rng = self.rng
        if self.last_msg is None or not self.note_texts:
            return
        side = rng.choice(["left", "right"])
        text = self.note_texts.pop()
        self.lines.append(f"note {side} : {text}" if rng.random() < 0.5 else f"note {side}: {text}")
        ends = sorted(set(self.last_msg), key=self.order.index)
        target = ends[0] if side == "left" else ends[-1]
        self.it.edge(self.it.node("note", text), target.nid, "anchor")

    def note_on(self, alive: list[_P]) -> None:
        rng = self.rng
        if not self.note_texts:
            return
        text = self.note_texts.pop()
        if len(alive) >= 2 and rng.random() < 0.3:
            a, b = rng.sample(alive, 2)
            targets, head = [a, b], f"note over {a.code}, {b.code}"
        else:
            a = rng.choice(alive)
            pos = rng.choice(["left of", "right of", "over", "over"])
            kw = rng.choices(["note", "hnote", "rnote"], weights=(80, 10, 10))[0] if pos == "over" else "note"
            targets, head = [a], f"{kw} {pos} {a.code}"
        parts = text.split(" ", 2)
        if len(parts) >= 3 and rng.random() < 0.25:  # block note on two lines
            lines = [" ".join(parts[:2]), parts[2]]
            text = "\n".join(lines)
            self.lines += [head] + [f"  {x}" for x in lines] + [f"end {head.split()[0]}"]
        else:
            self.lines.append(f"{head} : {text}")
        nid = self.it.node("note", text)
        for t in targets:
            self.touch(t)
            self.it.edge(nid, t.nid, "anchor")

    def alive(self) -> list[_P]:
        return [p for p in self.parts if not p.dead and (not p.created or p.nid)]

    def call(self, src: _P, depth: int) -> None:
        """src calls someone; the callee may call on, then usually replies."""
        rng = self.rng
        cands = [p for p in self.alive() if p is not src and not (p.created and not p.nid)]
        if not cands:
            return
        tgt = rng.choice(cands)
        act = self.act
        if act in ("shorthand", "return"):
            self.msg(src, tgt, "sync", self.label("sync", src, tgt), mods="++")
        else:
            self.msg(src, tgt, "sync", self.label("sync", src, tgt))
            if act == "explicit":
                self.lines.append(f"activate {tgt.code}")
        if self.rng.random() < 0.08:
            self.note_after_message()
        if depth < 2 and self.budget > 2 and rng.random() < 0.45:
            for _ in range(rng.choice([1, 1, 2])):
                if self.budget <= 1:
                    break
                if rng.random() < 0.2:
                    self.msg(tgt, tgt, "sync", self.label("self", tgt, tgt))
                else:
                    self.call(tgt, depth + 1)
        reply = act == "return" or rng.random() < 0.72
        if act == "return":
            label = self.label("reply", tgt, src)
            self.lines.append(f"return {label}".rstrip())
            self.emit_edge(tgt, src, "reply", label)
        elif reply:
            mods = "--" if act == "shorthand" else ""
            self.msg(tgt, src, "reply", self.label("reply", tgt, src), mods=mods)
        elif act == "shorthand":
            self.lines.append(f"deactivate {tgt.code}")
        if act == "explicit":
            self.lines.append(f"deactivate {tgt.code}")

    def step(self, depth: int) -> None:
        """One interaction at the current position."""
        rng = self.rng
        alive = self.alive()
        r = rng.random()
        src = alive[0] if (alive[0].kind == "actor" and rng.random() < 0.45) else rng.choice(alive)
        if r < 0.62:
            self.call(src, depth)
        elif r < 0.78:
            tgts = [p for p in alive if p is not src]
            if tgts:
                self.msg(src, rng.choice(tgts), "async", self.label("async", src, src))
        elif r < 0.9:
            self.msg(src, src, "sync", self.label("self", src, src))
        else:
            self.note_on(alive)
            return
        if rng.random() < 0.06:
            self.note_after_message()

    def block(self, depth: int, quota: int) -> None:
        """Emit about `quota` messages (at least one) at this nesting level."""
        target = self.budget - quota
        start, guard = self.budget, 0
        while (self.budget == start or self.budget > max(target, 0)) and guard < 60:
            guard += 1
            if depth < 2 and self.budget >= 3 and self.rng.random() < (0.22 if depth == 0 else 0.1):
                self.fragment(depth)
            else:
                before = self.budget
                self.step(depth)
                if self.budget == before and self.rng.random() < 0.5:  # a note only
                    continue
            if self.rng.random() < 0.04:
                self.gap = self.rng.choice(["...", "|||"])

    def fragment(self, depth: int) -> None:
        rng, dom = self.rng, self.dom
        kind = rng.choices(["alt", "opt", "loop", "par", "break", "critical", "group"],
                           weights=(30, 16, 22, 10, 7, 5, 10))[0]
        cond = rng.choice(dom.checks).rstrip("?").lower()
        if kind == "alt":
            guards = [cond] + [rng.choice(ELSE_GUARDS) for _ in range(rng.choice([1, 1, 1, 2]))]
            if rng.random() < 0.1:
                guards = guards[:1]
        elif kind == "par":
            guards = [""] * rng.choice([2, 2, 3])
        elif kind == "loop":
            guards = [rng.choice(LOOP_GUARDS)]
        elif kind == "group":
            guards = [rng.choice(GROUP_LABELS)]
        elif kind == "critical":
            guards = [rng.choice(CRITICAL_GUARDS)]
        else:
            guards = [cond]
        parent = self.holders[-1].id if self.holders else None
        multi = len(guards) > 1
        if multi:
            top = self.it.group(kind, "", parent)
        for k, guard in enumerate(guards):
            shown = guard if kind == "group" else (f"[{guard}]" if guard else "")
            if k == 0:
                self.lines.append(f"{kind} {guard}".rstrip())
            else:
                self.lines.append(f"else {guard}".rstrip())
            holder = self.it.group("operand", shown, top.id) if multi else self.it.group(kind, shown, parent)
            self.holders.append(holder)
            self.block(depth + 1, max(1, min(self.budget - 1, rng.choice([1, 1, 2, 2, 3]))))
            self.holders.pop()
        self.lines.append("end")

    def lifecycle(self) -> None:
        """Maybe one participant created mid-flow (and maybe destroyed at the end of its use)."""
        rng = self.rng
        name = rng.choice(self.dom.entities)
        if any(p.text.lower() == name.lower() or p.code.lower() == name.lower() for p in self.parts) \
                or is_keyword(name):
            return
        p = _P(code=name, text=name, kind=rng.choice(["participant", "entity", "control"]), created=True)
        creator = rng.choice([q for q in self.alive() if q.kind not in ("database", "queue", "collections")]
                             or self.alive())
        self.parts.append(p)
        self.gap = None
        kw = "" if p.kind == "participant" and rng.random() < 0.5 else f"{p.kind} "
        self.lines.append(f"create {kw}{p.code}")
        p.nid = self.it.node(p.kind, p.text)
        self.order.append(p)
        arg = camel(words(rng.choice(self.dom.fields)))
        label = rng.choice(["new()", f"new({arg})", "create", "instantiate", "build()"])
        self.msg(creator, p, "create", label)
        self.budget += 1  # the creation message does not use the budget
        if rng.random() < 0.4:
            label = rng.choice(["close()", "dispose()", "delete", "release()", "finish"])
            self.msg(creator, p, "destroy", label)
            self.lines.append(f"destroy {p.code}")
            p.dead = True
            self.budget += 1


def generate(rng: random.Random, dom: Domain, direction: Optional[str] = None,
             colors: bool = True) -> tuple[list[str], Diagram]:
    s = _Seq(rng, dom, colors)
    s.cast()
    if rng.random() < 0.1:
        s.lines.append("autonumber")
    s.declare_all()
    s.budget = rng.choices(range(3, 26), weights=[3, 6, 8, 9, 10, 10, 9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2,
                                                 2, 1, 1, 1, 1])[0]
    lifecycle_at = rng.randint(1, max(1, s.budget - 2)) if rng.random() < 0.12 else -1
    ref_at = rng.randint(1, max(1, s.budget - 1)) if rng.random() < 0.05 else -1
    while s.budget > 0:
        before = s.budget
        if s.budget == lifecycle_at:
            s.lifecycle()
            lifecycle_at = -1
        if s.budget == ref_at:
            alive = s.alive()
            who = rng.sample(alive, min(len(alive), 2))
            for p in who:
                s.touch(p)
            text = rng.choice(GROUP_LABELS + tuple(t.lower() for t in dom.tasks[:5]))
            s.lines.append(f"ref over {', '.join(p.code for p in who)} : {text}")
            g = s.it.group("ref", text)
            g.members = [p.nid for p in who]
            ref_at = -1
        s.block(0, rng.choice([1, 2, 3]))
        if s.budget == before:
            s.budget -= 1
    return s.lines, s.it.d
