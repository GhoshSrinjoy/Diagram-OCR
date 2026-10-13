"""Canonical form: one unique serialization per diagram, used for training targets.

Without it the same picture has many valid answers (any node naming, any line order),
and the model is penalized for harmless differences - v1 lost loss on `D` vs `E` renames.

Rules (SPEC.md, "Canonical form"):
  * Flow types (flowchart, activity, state, graph, circuit): nodes are ordered by
    following the arrows - a multi-source BFS from the entry nodes (no incoming arrow;
    start/initial/input kinds first), visiting neighbours in reading order.
  * Layout types (class, er, sequence): nodes in reading order (rows top to bottom,
    left to right inside a row).
  * Without coordinates, branches follow the order edges were written and other
    ties fall back to the original node order.
  * Ids become n1..nN / g1..gM; edges are sorted by (source, target) except in
    sequence diagrams, where line order is time order and is kept.
  * Whitespace inside texts is collapsed; explicit line breaks are kept.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict, deque
from typing import Optional

from .model import EDGE_ID_RE, Diagram, Group, edge_id

FLOW_TYPES = frozenset({"flowchart", "activity", "state", "graph", "circuit"})
START_KINDS = frozenset({"terminal", "initial", "input"})
ROW_OVERLAP = 0.5  # min vertical overlap (fraction of the smaller box) to share a reading row


def normalize_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _opt_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = normalize_text(text)
    return t or None


def reading_rank(boxes: list[tuple[int, int, int, int]]) -> list[int]:
    """Rank of each box in reading order: rows top to bottom, left to right in a row.

    Boxes join the current row when they overlap its first box vertically by at least
    ROW_OVERLAP of the smaller height - centre distance alone fails for rows of boxes
    with different heights (UML classes)."""
    by_top = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0], i))
    rows: list[list[int]] = []
    anchor = (0, 0)
    for i in by_top:
        y1, y2 = boxes[i][1], boxes[i][3]
        if rows:
            overlap = min(y2, anchor[1]) - max(y1, anchor[0])
            if overlap >= 0 and overlap >= ROW_OVERLAP * max(min(y2 - y1, anchor[1] - anchor[0]), 1):
                rows[-1].append(i)
                continue
        rows.append([i])
        anchor = (y1, y2)
    rank = [0] * len(boxes)
    pos = 0
    for row in rows:
        for i in sorted(row, key=lambda j: (boxes[j][0] + boxes[j][2], j)):
            rank[i] = pos
            pos += 1
    return rank


def _group_entries(d: Diagram, rank: list[int]) -> dict[str, list[int]]:
    """Where an arrow into a container enters it, chosen by structure only (never by how the
    member list happens to be ordered, which would break idempotency): the container's
    `initial` nodes, else members without an incoming arrow from inside it, else all members.
    Containers without direct node members delegate to their child containers."""
    node_idx = {n.id: i for i, n in enumerate(d.nodes)}
    children: dict[str, list[str]] = defaultdict(list)
    for g in d.groups:
        if g.parent:
            children[g.parent].append(g.id)
    by_id = {g.id: g for g in d.groups}

    def entries(gid: str, seen: set[str]) -> list[int]:
        if gid in seen or gid not in by_id:
            return []
        seen.add(gid)
        direct = {node_idx[m] for m in by_id[gid].members if m in node_idx}
        if not direct:
            return sorted({i for c in children[gid] for i in entries(c, seen)}, key=lambda i: rank[i])
        found = [i for i in direct if d.nodes[i].kind == "initial"]
        if not found:
            entered = {node_idx[e.target] for e in d.edges
                       if e.directed and e.kind != "bidir" and e.source != e.target
                       and node_idx.get(e.source) in direct and node_idx.get(e.target) in direct}
            found = [i for i in direct if i not in entered] or list(direct)
        return sorted(found, key=lambda i: rank[i])

    return {g.id: es for g in d.groups if (es := entries(g.id, set()))}


def _flow_order(d: Diagram, rank: list[int], layout: bool) -> list[int]:
    n = len(d.nodes)
    node_idx = {node.id: i for i, node in enumerate(d.nodes)}
    group_entries = _group_entries(d, rank)

    def resolve(ref: str) -> list[int]:
        return [node_idx[ref]] if ref in node_idx else group_entries.get(ref, [])

    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)  # u -> [(edge index, v)]
    indeg = [0] * n
    for ei, e in enumerate(d.edges):
        for s in resolve(e.source):
            for t in resolve(e.target):
                if s == t:
                    continue
                adj[s].append((ei, t))
                if e.directed and e.kind != "bidir":
                    indeg[t] += 1
                else:  # undirected / double-headed: walkable both ways
                    adj[t].append((ei, s))

    is_note = [node.kind == "note" for node in d.nodes]
    order: list[int] = []
    seen = [False] * n

    def bfs(starts: list[int]) -> None:
        queue: deque[int] = deque()
        for s in starts:
            if not seen[s]:
                seen[s] = True
                queue.append(s)
        while queue:
            u = queue.popleft()
            order.append(u)
            # Branches in reading order when we know the layout, else in the order the
            # edges were written (the author's order, e.g. Mermaid source).
            for _, v in sorted(adj[u], key=lambda p: (rank[p[1]], p[0]) if layout else (p[0],)):
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)

    # Connected pieces one after another (main flow first, strays later); inside a piece all
    # its entry nodes start together (e.g. every input of one circuit).
    comp = list(range(n))

    def find(x: int) -> int:
        while comp[x] != x:
            comp[x] = comp[comp[x]]
            x = comp[x]
        return x

    for u in range(n):
        for _, v in adj[u]:
            comp[find(u)] = find(v)

    def root_key(i: int) -> tuple:
        return (d.nodes[i].kind not in START_KINDS, rank[i])

    by_comp: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if indeg[i] == 0 and not is_note[i]:
            by_comp[find(i)].append(i)
    for roots in sorted(by_comp.values(), key=lambda rs: min(root_key(i) for i in rs)):
        bfs(sorted(roots, key=root_key))
    while len(order) < n:  # cycles with no entry, notes, ...
        rest = [i for i in range(n) if not seen[i]]
        bfs([min(rest, key=lambda i: (is_note[i], rank[i]))])
    return order


def canonicalize(d: Diagram) -> Diagram:
    d = copy.deepcopy(d)
    n = len(d.nodes)

    # 1. texts
    d.title = _opt_text(d.title)
    for node in d.nodes:
        node.text = normalize_text(node.text)
        node.stereotype = _opt_text(node.stereotype)
        for m in node.members:
            m.text = normalize_text(m.text)
    for e in d.edges:
        e.label = normalize_text(e.label)
        e.source_label = _opt_text(e.source_label)
        e.target_label = _opt_text(e.target_label)
        if e.style == "solid":
            e.style = None
    for g in d.groups:
        g.text = normalize_text(g.text)

    # 2. node order
    layout = d.has_layout()
    if layout:
        rank = reading_rank([node.bbox for node in d.nodes])  # type: ignore[misc]
    else:
        rank = list(range(n))  # original order is the only tie-breaker we have
    if d.type in FLOW_TYPES:
        order = _flow_order(d, rank, layout)
    else:
        order = sorted(range(n), key=lambda i: rank[i])
    node_map = {d.nodes[old].id: f"n{new + 1}" for new, old in enumerate(order)}
    d.nodes = [d.nodes[i] for i in order]
    node_pos = {node.id: i for i, node in enumerate(d.nodes)}  # still old ids here

    # 3. group order: parents first, then by earliest member (incl. descendants)
    children: dict[str, list[Group]] = defaultdict(list)
    for g in d.groups:
        if g.parent:
            children[g.parent].append(g)

    def member_rank(ref: str) -> float:
        if ref in node_pos:
            return node_pos[ref]
        if EDGE_ID_RE.match(ref):
            return n + int(ref[1:])  # original edge order; edges never outrank nodes
        return float("inf")

    def min_rank(g: Group, seen: set[str]) -> float:
        if g.id in seen:
            return float("inf")
        seen.add(g.id)
        own = min((member_rank(m) for m in g.members), default=float("inf"))
        sub = min((min_rank(c, seen) for c in children[g.id]), default=float("inf"))
        return min(own, sub)

    def depth(g: Group) -> int:
        k, cur, by_id = 0, g.parent, {x.id: x for x in d.groups}
        while cur in by_id and k <= len(d.groups):
            k, cur = k + 1, by_id[cur].parent
        return k

    group_order = sorted(range(len(d.groups)),
                         key=lambda i: (min_rank(d.groups[i], set()), depth(d.groups[i]), i))
    group_map = {d.groups[old].id: f"g{new + 1}" for new, old in enumerate(group_order)}
    d.groups = [d.groups[i] for i in group_order]

    # 4. edge order. A container endpoint sorts right after the node where the flow enters it,
    # so its edges sit in the same block the traversal reads them from (keeps idempotency).
    entry_pos = {gid: min(es) for gid, es in _group_entries(d, list(range(n))).items()}

    def ref_pos(ref: str) -> float:
        if ref in node_pos:
            return node_pos[ref]
        if ref in entry_pos:
            return entry_pos[ref] + 0.5
        if ref in group_map:
            return n + int(group_map[ref][1:])
        return float("inf")

    # Undirected and double-headed edges have no inherent orientation: store them from the
    # earlier endpoint to the later one, so the same picture always gives the same line (and
    # canonicalize stays idempotent - flow order walks these edges in stored order).
    for e in d.edges:
        if (not e.directed or e.kind == "bidir") and ref_pos(e.source) > ref_pos(e.target):
            e.source, e.target = e.target, e.source
            e.source_label, e.target_label = e.target_label, e.source_label
            e.source_port, e.target_port = e.target_port, e.source_port
            if e.path:
                e.path = list(reversed(e.path))

    if d.type == "sequence":
        edge_order = list(range(len(d.edges)))
    else:
        edge_order = sorted(
            range(len(d.edges)),
            key=lambda i: (ref_pos(d.edges[i].source), ref_pos(d.edges[i].target),
                           d.edges[i].kind, d.edges[i].label, i),
        )
    edge_map = {edge_id(old): edge_id(new) for new, old in enumerate(edge_order)}
    d.edges = [d.edges[i] for i in edge_order]

    # 5. rename everything
    ref_map = {**node_map, **group_map}
    for node in d.nodes:
        node.id = node_map[node.id]
    for e in d.edges:
        e.source = ref_map.get(e.source, e.source)
        e.target = ref_map.get(e.target, e.target)
    new_pos = {f"n{i + 1}": i for i in range(n)}
    for g in d.groups:
        g.id = group_map[g.id]
        g.parent = ref_map.get(g.parent, g.parent) if g.parent else None
        renamed = []
        for m in g.members:
            r = edge_map.get(m, m) if EDGE_ID_RE.match(m) else ref_map.get(m, m)
            if r not in renamed:
                renamed.append(r)
        nodes_first = [r for r in renamed if r in new_pos]
        edges_after = [r for r in renamed if EDGE_ID_RE.match(r)]
        others = [r for r in renamed if r not in new_pos and not EDGE_ID_RE.match(r)]
        g.members = (sorted(nodes_first, key=new_pos.__getitem__)
                     + sorted(edges_after, key=lambda r: int(r[1:])) + others)
    return d
