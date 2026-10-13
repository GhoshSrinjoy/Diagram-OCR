"""Synthetic PlantUML component and deployment diagrams: components (declared or `[Name]`
shorthand), interfaces, nodes, databases, clouds, queues, folders, artifacts, storage, frames and
packages, nested containers, and links (solid / dashed, one-way / two-way / plain) with protocol
labels. The importer maps them to the generic `graph` type (element keyword -> node kind,
container -> cluster). `generate` returns the PlantUML body and that Diagram.

Not used: notes (the component importer drops them, so they would be drawn but not in the
target), stereotypes and `<<...>>` / `>` labels (not kept as drawn), `()` interface declarations
(handled by a different parser path)."""

from __future__ import annotations

import random
import re
from typing import Optional

from diagram_ir import Diagram, Group

from .puml_check import Intent
from .puml_words import (
    ARTIFACTS,
    CLOUDS,
    DEVICE_NODES,
    FOLDERS,
    INFRA_NODES,
    PROTOCOLS,
    Domain,
    ident,
    is_keyword,
    words,
)

KIND = {"component": "component", "interface": "circle", "node": "box3d", "database": "cylinder",
        "cloud": "cloud", "folder": "folder", "artifact": "file", "queue": "queue", "storage": "storage",
        "frame": "box", "rectangle": "box", "package": "folder", "actor": "actor", "file": "file",
        "card": "box", "hexagon": "hexagon"}
CONTAINERS = ("package", "node", "frame", "rectangle", "cloud", "folder")
LAYERS = ("Frontend", "Backend", "Data Layer", "Integration", "Core", "Edge", "Services", "Persistence",
          "Messaging", "Presentation", "Security", "Analytics")
QUEUES = ("Job Queue", "Order Queue", "Event Bus", "Dead Letter Queue", "Task Queue", "Message Broker",
          "Retry Queue", "Outbox")
VERB_LABELS = ("uses", "reads", "writes", "calls", "publishes", "subscribes", "queries", "notifies", "stores",
               "syncs", "sends events", "pulls config")


def _store_kw(text: str) -> str:
    """A data store's element keyword: event streams / queues are drawn as queues."""
    streamy = any(w in text.lower() for w in ("queue", "events", "stream", "bus"))
    return "queue" if streamy else "database"


class _E:
    def __init__(self, kw: str, text: str, code: str, shorthand: bool = False):
        self.kw, self.text, self.code, self.shorthand = kw, text, code, shorthand
        self.nid = ""
        self.group: Optional[int] = None


class _Gen:
    def __init__(self, rng: random.Random, dom: Domain, flavor: str):
        self.rng = rng
        self.dom = dom
        self.flavor = flavor
        self.taken: set[str] = set()
        self.elems: list[_E] = []
        self.containers: list[dict] = []  # {kw, text, code, parent, gid}

    def code_for(self, text: str) -> str:
        base = ident(text)
        code, k = base, 2
        while code.lower() in self.taken or is_keyword(code):
            code, k = f"{base}{k}", k + 1
        self.taken.add(code.lower())
        return code

    def add(self, kw: str, text: str, group: Optional[int] = None) -> Optional[_E]:
        if text.lower() in self.taken:
            return None
        shorthand = kw == "component" and group is None and self.rng.random() < 0.25 and "]" not in text
        if shorthand:
            e = _E(kw, text, f"[{text}]", True)
        elif text.isidentifier() and not is_keyword(text) and self.rng.random() < 0.5:
            e = _E(kw, text, text)
            self.taken.add(text.lower())
        else:
            e = _E(kw, text, self.code_for(text))
        self.taken.add(text.lower())
        e.group = group
        self.elems.append(e)
        return e

    def container(self, kw: str, text: str, parent: Optional[int] = None) -> int:
        code = self.code_for(text)
        self.taken.add(text.lower())
        self.containers.append({"kw": kw, "text": text, "code": code, "parent": parent})
        return len(self.containers) - 1

    def spaced(self, name: str) -> str:
        """`OrderService` -> sometimes "Order Service" (acronyms kept: `EHRSystem` -> "EHR System")."""
        if self.rng.random() >= 0.4:
            return name
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+", name)
        return " ".join(p if p.isupper() else p.capitalize() for p in parts) or name

    # ---- content ---------------------------------------------------------------------
    def component_view(self) -> None:
        rng, dom = self.rng, self.dom
        services = list(dom.services)
        rng.shuffle(services)
        n_comp = rng.randint(3, 9)
        n_cont = rng.choices([0, 1, 2, 3], weights=(20, 35, 30, 15))[0]
        layers = list(LAYERS)
        rng.shuffle(layers)
        conts = [self.container(rng.choice(CONTAINERS[:4]), layers[k]) for k in range(n_cont)]
        for k, name in enumerate(services[:n_comp]):
            g = conts[k % len(conts)] if conts and rng.random() < 0.8 else None
            self.add("component", self.spaced(name), g)
        for _ in range(rng.choices([0, 1, 2, 3], weights=(30, 35, 25, 10))[0]):
            base = rng.choice(services)
            text = rng.choice([f"I{base}", f"{words(base)[0].capitalize()} API", "REST API", "Events API",
                               "gRPC", "Admin API", f"{base} Port"])
            self.add("interface", text, rng.choice(conts) if conts and rng.random() < 0.3 else None)
        stores = list(dom.stores)
        rng.shuffle(stores)
        for text in stores[:rng.choices([0, 1, 2], weights=(25, 50, 25))[0]]:
            self.add(_store_kw(text), text, rng.choice(conts) if conts and rng.random() < 0.4 else None)
        if rng.random() < 0.15:
            self.add(rng.choice(["artifact", "file", "folder"]), rng.choice(ARTIFACTS + FOLDERS))

    def deployment_view(self) -> None:
        rng, dom = self.rng, self.dom
        infra = list(INFRA_NODES)
        rng.shuffle(infra)
        n_nodes = rng.randint(2, 5)
        nodes = [self.container("node", infra[k]) for k in range(n_nodes)]
        if rng.random() < 0.35:
            cl = self.container("cloud", rng.choice(CLOUDS))
            nodes.append(cl)
            if rng.random() < 0.4 and len(nodes) > 2:  # a node inside the cloud
                self.containers[nodes[1]]["parent"] = cl
        services = list(dom.services)
        rng.shuffle(services)
        stores = list(dom.stores)
        rng.shuffle(stores)
        for g in nodes:
            for _ in range(rng.choices([0, 1, 2, 3], weights=(10, 45, 35, 10))[0]):
                r = rng.random()
                if r < 0.45 and services:
                    self.add("component", self.spaced(services.pop()), g)
                elif r < 0.7:
                    self.add("artifact", rng.choice(ARTIFACTS), g)
                elif r < 0.9 and stores:
                    text = stores.pop()
                    self.add(_store_kw(text), text, g)
                elif rng.random() < 0.3:
                    self.add("queue", rng.choice(QUEUES), g)
                else:
                    self.add(rng.choice(["storage", "folder"]), rng.choice(FOLDERS), g)
        if rng.random() < 0.5:
            self.add("node", rng.choice(DEVICE_NODES))
        if rng.random() < 0.4 and stores:
            text = stores.pop()
            self.add(_store_kw(text), text)
        if rng.random() < 0.3:
            self.add("actor", rng.choice(dom.actors))
        if rng.random() < 0.15:
            self.add(rng.choice(["storage", "folder", "file"]), rng.choice(FOLDERS + ARTIFACTS))

    # ---- emission ---------------------------------------------------------------------
    def decl(self, e: _E) -> Optional[str]:
        if e.shorthand:  # only met in links: a bare `[X]` line would switch importer paths
            return None
        if e.code == e.text:
            return f"{e.kw} {e.code}"
        if self.rng.random() < 0.8:
            return f'{e.kw} "{e.text}" as {e.code}'
        return f'{e.kw} {e.code} as "{e.text}"'

    def block(self, idx: Optional[int], indent: str, out: list[str]) -> None:
        for e in self.elems:
            if e.group == idx:
                line = self.decl(e)
                if line:
                    out.append(indent + line)
        for k, c in enumerate(self.containers):
            if c["parent"] == idx:
                name = f'"{c["text"]}"' if (" " in c["text"] or self.rng.random() < 0.5) else c["text"]
                head = f'{c["kw"]} {name} as {c["code"]} {{' if name.startswith('"') else \
                    f'{c["kw"]} {c["text"]} {{'
                if not name.startswith('"'):
                    c["code"] = c["text"]
                out.append(indent + head)
                self.block(k, indent + "  ", out)
                out.append(indent + "}")


def generate(rng: random.Random, dom: Domain, direction: Optional[str] = "TB", colors: bool = True,
             flavor: str = "component") -> tuple[list[str], Diagram]:
    g = _Gen(rng, dom, flavor)
    if flavor == "deployment":
        g.deployment_view()
    else:
        g.component_view()
    # drop containers that ended up empty (an empty container is drawn as a plain element)
    while True:
        full = {e.group for e in g.elems} | {c["parent"] for c in g.containers}
        empty = [k for k in range(len(g.containers)) if k not in full]
        if not empty:
            break
        k = empty[-1]
        for c in g.containers:
            if c["parent"] is not None and c["parent"] > k:
                c["parent"] -= 1
        for e in g.elems:
            if e.group is not None and e.group > k:
                e.group -= 1
        del g.containers[k]
    if not any(e.kw == "component" or e.kw == "node" for e in g.elems) and not any(
            c["kw"] in ("node", "cloud", "folder", "frame") for c in g.containers):
        g.add("component", g.spaced(rng.choice(dom.services)) + " Core")  # keep it a component diagram

    out: list[str] = []
    g.block(None, "", out)
    it = Intent("graph", direction)
    gids: list[Group] = []
    for c in g.containers:
        gids.append(it.group("cluster", c["text"]))
    for k, c in enumerate(g.containers):
        if c["parent"] is not None:
            gids[k].parent = gids[c["parent"]].id
    for e in g.elems:
        e.nid = it.node(KIND[e.kw], e.text)
        if e.group is not None:
            gids[e.group].members.append(e.nid)

    # ---- links -----------------------------------------------------------------------
    elems = g.elems
    pairs: set[tuple[int, int]] = set()
    links: list[str] = []

    def link(a: _E, b: _E, kind: str, label: str, dashed: bool) -> None:
        key = (min(id(a), id(b)), max(id(a), id(b)))
        if a is b or key in pairs:
            return
        pairs.add(key)
        body = "." if dashed else "-"
        r = rng.random()
        if r < 0.65 or (label and r < 0.9):  # Graphviz drops labels of same-rank links: none there
            body *= 2
        elif r < 0.8:
            body = f"{body}{rng.choice(['up', 'down', 'left', 'right'])}{body}"
        elif r < 0.9 or label:
            body *= 3
        if kind == "arrow":
            line = f"{a.code} {body}> {b.code}" if rng.random() < 0.85 else f"{b.code} <{body} {a.code}"
        elif kind == "bidir":
            line = f"{a.code} <{body}> {b.code}"
        else:
            line = f"{a.code} {body} {b.code}"
        links.append(line + (f" : {label}" if label else ""))
        it.edge(a.nid, b.nid, kind, label, style="dashed" if dashed else None)

    ifaces = [e for e in elems if e.kw == "interface"]
    comps = [e for e in elems if e.kw in ("component", "artifact")]
    for e in ifaces:  # provided by one component, used by others
        if comps:
            link(rng.choice(comps), e, "line", "", False)
            for user in rng.sample(comps, min(len(comps), rng.choice([0, 1, 1, 2]))):
                link(user, e, "arrow", rng.choice(["", "", "uses"]), True)
    order = elems[:]
    rng.shuffle(order)
    for k, e in enumerate(order):  # connect everything (a `[X]` component exists only in links)
        if any(id(e) in p for p in pairs) or len(order) < 2:
            continue
        other = rng.choice(order[:k] or order[k + 1:])
        kind = rng.choices(["arrow", "line", "bidir"], weights=(70, 22, 8))[0]
        label = rng.choice(PROTOCOLS + VERB_LABELS) if rng.random() < 0.55 else ""
        link(other, e, kind, label, rng.random() < 0.25)
    for _ in range(rng.randint(0, max(0, len(elems) // 3))):
        a, b = rng.sample(elems, 2) if len(elems) >= 2 else (elems[0], elems[0])
        kind = rng.choices(["arrow", "line", "bidir"], weights=(70, 22, 8))[0]
        label = rng.choice(PROTOCOLS + VERB_LABELS) if rng.random() < 0.5 else ""
        link(a, b, kind, label, rng.random() < 0.25)
    if g.containers and rng.random() < 0.15:  # a link to a container itself
        c = rng.randrange(len(g.containers))
        e = rng.choice(elems)
        if e.group != c:
            links.append(f"{e.code} --> {g.containers[c]['code']}")
            it.edge(e.nid, gids[c].id, "arrow")
    rng.shuffle(links)
    return out + links, it.d
