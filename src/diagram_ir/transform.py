"""Structural transforms on a Diagram (used by importers and dataset builders)."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable

from .model import Diagram, Edge, Node, edge_id


def contract_nodes(d: Diagram, should_contract: Callable[[Node], bool]) -> Diagram:
    """Remove pass-through nodes (junction dots, routing points) and connect each predecessor
    directly to each successor, following chains of junctions (A -> S0 -> S1 -> B).

    The new edge takes kind / style / target end from the last segment (the one carrying the
    arrowhead) and source end from the first; its label is the last non-empty segment label.
    Edges are followed source -> target only. Group references to removed nodes and to the old
    edge ids are dropped (edge ids change)."""
    d = copy.deepcopy(d)
    drop = {n.id for n in d.nodes if should_contract(n)}
    if not drop:
        return d
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    for e in d.edges:
        outgoing[e.source].append(e)

    new_edges: list[Edge] = []
    seen_keys: set[tuple] = set()

    def walk(first: Edge, seg: Edge, visited: frozenset[str], labels: tuple[str, ...]) -> None:
        if seg.target not in drop:
            merged = copy.deepcopy(seg)
            merged.source = first.source
            merged.source_label, merged.source_port = first.source_label, first.source_port
            merged.label = next((lb for lb in reversed(labels + (seg.label,)) if lb), "")
            merged.style = seg.style or first.style
            key = (merged.source, merged.target, merged.kind, merged.label, merged.style)
            if key not in seen_keys:
                seen_keys.add(key)
                new_edges.append(merged)
            return
        if seg.target in visited:
            return  # junction cycle
        for nxt in outgoing[seg.target]:
            walk(first, nxt, visited | {seg.target}, labels + (seg.label,))

    for e in d.edges:
        if e.source not in drop:
            walk(e, e, frozenset(), ())

    old_edge_ids = {edge_id(i) for i in range(len(d.edges))}
    d.nodes = [n for n in d.nodes if n.id not in drop]
    d.edges = new_edges
    for g in d.groups:
        g.members = [m for m in g.members if m not in drop and m not in old_edge_ids]
    return d


def is_junction(n: Node) -> bool:
    """Unlabelled dots / tiny circles that only route lines."""
    return not n.text.strip() and n.kind in {"connector", "point", "circle", "junction"}
