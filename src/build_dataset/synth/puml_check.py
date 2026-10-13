"""Generator-vs-importer cross-check for the synthetic PlantUML data.

Every generator also builds the Diagram it means to draw (its "intent"). `compare` checks the
importer's result against that intent up to ids and element order:

  * nodes: multiset of (kind, text, stereotype, members, enclosing group chains);
  * edges: (kind, label, end labels, style, endpoints). Endpoint labels are refined a few rounds
    Weisfeiler-Lehman style, so text-free nodes (merge / fork / join / initial / final / choice)
    are told apart by their neighbourhood, not just their kind;
  * groups: (kind, text, parent chain, member nodes, member edges);
  * sequence diagrams: the edge list must also match in order (time order).

The result is None when both agree, else (category, detail) for the skip log.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from diagram_ir import Diagram, Edge, Group, Member, Node, normalize_text
from diagram_ir.model import UNDIRECTED_EDGE_KINDS

ROUNDS = 3


class Intent:
    """Builds the Diagram a generator means to draw (ids x1.., y1..; the importer's differ)."""

    def __init__(self, dtype: str, direction: Optional[str] = None):
        self.d = Diagram(type=dtype, direction=direction)

    def node(self, kind: str, text: str = "", stereotype: Optional[str] = None,
             members: Optional[list[tuple[str, str]]] = None) -> str:
        nid = f"x{len(self.d.nodes) + 1}"
        self.d.nodes.append(Node(id=nid, kind=kind, text=text, stereotype=stereotype or None,
                                 members=[Member(k, t) for k, t in members or []]))
        return nid

    def edge(self, source: str, target: str, kind: str, label: str = "", sl: Optional[str] = None,
             tl: Optional[str] = None, style: Optional[str] = None) -> str:
        self.d.edges.append(Edge(source=source, target=target, kind=kind, label=label, source_label=sl,
                                 target_label=tl, style=style))
        return f"e{len(self.d.edges)}"

    def group(self, kind: str, text: str = "", parent: Optional[str] = None) -> Group:
        g = Group(id=f"y{len(self.d.groups) + 1}", kind=kind, text=text, parent=parent)
        self.d.groups.append(g)
        return g


def _chain(gid: Optional[str], names: dict[str, tuple], parent: dict[str, Optional[str]]) -> tuple:
    out = []
    while gid is not None and gid in names and len(out) < 32:
        out.append(names[gid])
        gid = parent.get(gid)
    return tuple(out)


def _base_labels(d: Diagram) -> dict[str, str]:
    names = {g.id: (g.kind, normalize_text(g.text)) for g in d.groups}
    parent = {g.id: g.parent for g in d.groups}
    chains: dict[str, list[tuple]] = defaultdict(list)
    for g in d.groups:
        for m in g.members:
            chains[m].append(_chain(g.id, names, parent))
    lab = {}
    for n in d.nodes:
        members = tuple((m.kind, normalize_text(m.text)) for m in n.members)
        lab[n.id] = repr((n.kind, normalize_text(n.text), n.stereotype or "", members,
                          tuple(sorted(chains[n.id]))))
    for g in d.groups:
        lab[g.id] = repr(("group",) + _chain(g.id, names, parent))
    return lab


def _edge_key(e: Edge) -> tuple:
    return e.kind, normalize_text(e.label), e.style or ""


def _refine(d: Diagram, lab: dict[str, str]) -> dict[str, str]:
    for _ in range(ROUNDS):
        nbr: dict[str, list[tuple]] = defaultdict(list)
        for e in d.edges:
            sl, tl = e.source_label or "", e.target_label or ""
            if e.kind in UNDIRECTED_EDGE_KINDS:
                nbr[e.source].append(("u", _edge_key(e), sl, tl, lab.get(e.target, "?")))
                nbr[e.target].append(("u", _edge_key(e), tl, sl, lab.get(e.source, "?")))
            else:
                nbr[e.source].append(("o", _edge_key(e), sl, tl, lab.get(e.target, "?")))
                nbr[e.target].append(("i", _edge_key(e), sl, tl, lab.get(e.source, "?")))
        lab = {k: f"{hash((v, tuple(sorted(map(repr, nbr[k])))))}" for k, v in lab.items()}
    return lab


def _edge_sig(e: Edge, lab: dict[str, str]) -> tuple:
    a, b = (lab.get(e.source, "?"), e.source_label or ""), (lab.get(e.target, "?"), e.target_label or "")
    if e.kind in UNDIRECTED_EDGE_KINDS:
        a, b = sorted((a, b))
    return _edge_key(e) + (a, b)


def _signature(d: Diagram, ordered: bool) -> tuple[Counter, object, Counter]:
    lab = _refine(d, _base_labels(d))
    nodes = Counter(lab[n.id] for n in d.nodes)
    sigs = [_edge_sig(e, lab) for e in d.edges]
    edges: object = sigs if ordered else Counter(sigs)
    names = {g.id: (g.kind, normalize_text(g.text)) for g in d.groups}
    parent = {g.id: g.parent for g in d.groups}
    groups: Counter = Counter()
    for g in d.groups:
        node_members = sorted(lab[m] for m in g.members if m in lab)
        edge_members = [int(m[1:]) - 1 for m in g.members if m not in lab]
        if ordered:
            em: tuple = tuple(sorted(edge_members))
        else:
            em = tuple(sorted(repr(sigs[i]) for i in edge_members if 0 <= i < len(sigs)))
        groups[(_chain(g.id, names, parent), tuple(node_members), em)] += 1
    return nodes, edges, groups


def _readable_nodes(d: Diagram) -> Counter:
    return Counter(f"{n.kind}:{normalize_text(n.text)[:24]}" for n in d.nodes)


def _readable_edges(d: Diagram) -> Counter:
    text = {n.id: normalize_text(n.text)[:16] or n.kind for n in d.nodes}
    text.update({g.id: f"[{normalize_text(g.text)[:16]}]" for g in d.groups})
    out: Counter = Counter()
    for e in d.edges:
        a, b = text.get(e.source, "?"), text.get(e.target, "?")
        if e.kind in UNDIRECTED_EDGE_KINDS:
            a, b = sorted((a, b))
        out[f"{a}-{e.kind}->{b}:{normalize_text(e.label)[:16]}"] += 1
    return out


def _diff(want: Counter, got: Counter) -> str:
    missing = list((want - got).elements())[:3]
    extra = list((got - want).elements())[:3]
    return f"missing {missing} extra {extra}"


def compare(intent: Diagram, got: Diagram) -> Optional[tuple[str, str]]:
    """None if the importer's diagram matches the generator's intent, else (category, detail)."""
    if intent.type != got.type:
        return "type", f"{intent.type} -> {got.type}"
    if (intent.direction or None) != (got.direction or None):
        return "direction", f"{intent.direction} -> {got.direction}"
    ordered = intent.type == "sequence"
    n1, e1, g1 = _signature(intent, ordered)
    n2, e2, g2 = _signature(got, ordered)
    if n1 != n2:
        if len(intent.nodes) != len(got.nodes) or _readable_nodes(intent) != _readable_nodes(got):
            return "nodes", f"{len(intent.nodes)} vs {len(got.nodes)}: " + _diff(_readable_nodes(intent),
                                                                                 _readable_nodes(got))
        return "nodes (attributes / grouping / wiring)", _diff(_readable_edges(intent), _readable_edges(got))
    if e1 != e2:
        detail = _diff(_readable_edges(intent), _readable_edges(got))
        if ordered and _readable_edges(intent) == _readable_edges(got):
            detail = "same messages, different order"
        return "edges", f"{len(intent.edges)} vs {len(got.edges)}: {detail}"
    if g1 != g2:
        want = Counter(f"{k[0][:2]}" for k in g1.elements())
        have = Counter(f"{k[0][:2]}" for k in g2.elements())
        return "groups", f"{len(intent.groups)} vs {len(got.groups)}: {_diff(want, have)}"
    return None
