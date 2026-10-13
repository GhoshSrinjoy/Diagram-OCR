"""Graph-level metrics for comparing a predicted diagram with the gold one.

Node correspondence is found with a maximum-weight assignment (Hungarian) on a similarity that
mixes text (normalized edit similarity) and kind; for text-free nodes (logic gates, DiDi shape
drawings) kind, degree and position break the ties. Identical-looking nodes (same kind and text)
then trade partners while that makes more edges agree, since degree and position often pair them
wrongly. Groups are matched the same way (label and kind; shared members when unlabeled), so edges
that end at a container (a collapsed BPMN pool, a composite state) can be scored. Everything
downstream is measured through that matching:

    node_p / node_r / node_f1     matched node pairs (similarity >= threshold)
    text_exact                    matched pairs whose normalized text is identical
    kind_acc                      matched pairs with the same kind
    edge_p / edge_r / edge_f1     edges whose endpoints map onto a gold edge (direction-aware,
                                  either direction for undirected kinds)
    edge_label_f1                 same, also requiring the normalized label to match
    valid                         prediction parsed without syntax errors
    exact                         canonical DGM strings identical

`score(pred_text, gold_text)` takes raw model output (lenient parse) and returns a dict;
`aggregate(list_of_dicts)` micro-averages counts over a dataset.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Optional

from .canonical import canonicalize
from .dsl import dumps, parse
from .model import Diagram

MATCH_THRESHOLD = 0.5
# 2: edge-aware pairing of identical-looking nodes; edges ending at groups can match.
VERSION = 2

try:  # scipy is optional; fall back to greedy matching
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def text_similarity(a: str, b: str) -> float:
    a, b = norm_text(a), norm_text(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _degrees(d: Diagram) -> dict[str, int]:
    deg = {n.id: 0 for n in d.nodes}
    for e in d.edges:
        for ref in (e.source, e.target):
            if ref in deg:
                deg[ref] += 1
    return deg


def _assign(sim: list[list[float]], pred_ids: list[str], gold_ids: list[str],
            threshold: float) -> dict[str, str]:
    """Maximum-weight one-to-one assignment, keeping pairs whose similarity reaches `threshold`."""
    if not pred_ids or not gold_ids:
        return {}
    pairs: list[tuple[int, int]] = []
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment([[-v for v in row] for row in sim])
        pairs = list(zip(rows.tolist(), cols.tolist(), strict=True))
    else:
        used_i, used_j = set(), set()
        for _v, i, j in sorted(((sim[i][j], i, j) for i in range(len(sim)) for j in range(len(sim[0]))),
                              reverse=True):
            if i not in used_i and j not in used_j:
                used_i.add(i)
                used_j.add(j)
                pairs.append((i, j))
    return {pred_ids[i]: gold_ids[j] for i, j in pairs if sim[i][j] >= threshold}


def _node_similarity(pred: Diagram, gold: Diagram) -> list[list[float]]:
    pdeg, gdeg = _degrees(pred), _degrees(gold)
    sim = [[0.0] * len(gold.nodes) for _ in pred.nodes]
    for i, p in enumerate(pred.nodes):
        for j, g in enumerate(gold.nodes):
            kind = 1.0 if p.kind == g.kind else 0.0
            if norm_text(p.text) or norm_text(g.text):
                s = 0.85 * text_similarity(p.text, g.text) + 0.15 * kind
            else:  # both text-free: kind, then similar degree, then position in canonical order
                deg = 1.0 / (1.0 + abs(pdeg[p.id] - gdeg[g.id]))
                order = 1.0 - abs(i / len(pred.nodes) - j / len(gold.nodes))
                s = 0.6 * kind + 0.25 * deg + 0.15 * order
            sim[i][j] = s
    return sim


def _group_similarity(pred: Diagram, gold: Diagram, nodes: dict[str, str]) -> list[list[float]]:
    gold_node_ids = {n.id for n in gold.nodes}
    sim = [[0.0] * len(gold.groups) for _ in pred.groups]
    for i, p in enumerate(pred.groups):
        mapped = {nodes[m] for m in p.members if m in nodes}
        for j, g in enumerate(gold.groups):
            kind = 1.0 if p.kind == g.kind else 0.0
            if norm_text(p.text) or norm_text(g.text):
                s = 0.85 * text_similarity(p.text, g.text) + 0.15 * kind
            else:  # unlabeled containers: kind, then shared member nodes
                members = {m for m in g.members if m in gold_node_ids}
                union = mapped | members
                s = 0.6 * kind + 0.4 * (len(mapped & members) / len(union) if union else 0.0)
            sim[i][j] = s
    return sim


def match(pred: Diagram, gold: Diagram, threshold: float = MATCH_THRESHOLD
          ) -> tuple[dict[str, str], dict[str, str]]:
    """(node mapping, group mapping), each pred id -> gold id."""
    nodes = _assign(_node_similarity(pred, gold), [n.id for n in pred.nodes], [n.id for n in gold.nodes],
                    threshold)
    groups = _assign(_group_similarity(pred, gold, nodes), [g.id for g in pred.groups],
                     [g.id for g in gold.groups], threshold)
    return _refine(pred, gold, nodes, groups), groups


def match_nodes(pred: Diagram, gold: Diagram, threshold: float = MATCH_THRESHOLD) -> dict[str, str]:
    """Map pred node id -> gold node id."""
    return match(pred, gold, threshold)[0]


def _hit_change(counts: Counter, gold: Counter, removed: list, added: list) -> int:
    """Change in sum(min(counts[k], gold[k])) when `removed` keys leave and `added` keys arrive."""
    change: Counter = Counter()
    for k in removed:
        if k is not None:
            change[k] -= 1
    for k in added:
        if k is not None:
            change[k] += 1
    return sum(min(counts[k] + c, gold[k]) - min(counts[k], gold[k]) for k, c in change.items() if c)


def _refine(pred: Diagram, gold: Diagram, nodes: dict[str, str], groups: dict[str, str]) -> dict[str, str]:
    """Let identical-looking nodes (same kind and normalized text, e.g. text-free logic gates) trade
    gold partners while that makes more edges agree, then more edge labels. Only nodes of one such
    class trade places, so node, text and kind counts stay exactly as the assignment left them."""
    cls_p = {n.id: (n.kind, norm_text(n.text)) for n in pred.nodes}
    cls_g = {n.id: (n.kind, norm_text(n.text)) for n in gold.nodes}
    by_p: dict[tuple, list[str]] = defaultdict(list)
    by_g: dict[tuple, list[str]] = defaultdict(list)
    for nid, c in cls_p.items():
        by_p[c].append(nid)
    for nid, c in cls_g.items():
        by_g[c].append(nid)
    classes = [c for c in by_p if by_g.get(c) and len(by_p[c]) + len(by_g[c]) > 2]
    if not classes or not pred.edges or not gold.edges:
        return nodes

    ends = {**groups, **nodes}  # pred id -> gold id for every possible edge endpoint
    inv = {g: p for p, g in nodes.items()}
    edges = [(e.source, e.target, not e.directed or e.kind == "bidir", norm_text(e.label))
             for e in pred.edges]
    incident: dict[str, set[int]] = defaultdict(set)
    for i, (s, t, _, _) in enumerate(edges):
        incident[s].add(i)
        incident[t].add(i)

    def keys(i: int) -> tuple[Optional[tuple], Optional[tuple]]:
        s, t, undirected, label = edges[i]
        s, t = ends.get(s), ends.get(t)
        if s is None or t is None:
            return None, None
        if undirected:
            s, t = sorted((s, t))
        return (s, t), (s, t, label)

    gold_plain = Counter(_edge_keys(gold, None, False))
    gold_label = Counter(_edge_keys(gold, None, True))
    plain: Counter = Counter()
    labeled: Counter = Counter()
    for i in range(len(edges)):
        k, kl = keys(i)
        if k is not None:
            plain[k] += 1
            labeled[kl] += 1

    def assign(p: Optional[str], g: Optional[str]) -> None:
        if p is None:
            return
        if g is None:
            ends.pop(p, None)
        else:
            ends[p] = g

    def try_move(a: str, g: str) -> bool:
        """Give gold node g to pred node a; g's current holder (if any) takes a's old partner."""
        old, b = ends.get(a), inv.get(g)
        touched = incident[a] | (incident[b] if b is not None else set())
        before = [keys(i) for i in touched]
        assign(a, g)
        assign(b, old)
        after = [keys(i) for i in touched]
        gain = (_hit_change(plain, gold_plain, [k for k, _ in before], [k for k, _ in after]),
                _hit_change(labeled, gold_label, [k for _, k in before], [k for _, k in after]))
        if gain <= (0, 0):
            assign(a, old)
            assign(b, g)
            return False
        for counts, idx in ((plain, 0), (labeled, 1)):
            for k in before:
                if k[idx] is not None:
                    counts[k[idx]] -= 1
            for k in after:
                if k[idx] is not None:
                    counts[k[idx]] += 1
        inv[g] = a
        if old is not None and b is not None:
            inv[old] = b
        elif old is not None:
            del inv[old]
        return True

    improved = True
    while improved:  # every accepted move adds an edge (or label) hit, so this ends
        improved = False
        for c in classes:
            for a in by_p[c]:
                if a in ends and cls_g[ends[a]] != c:
                    continue  # paired across classes (fuzzy text match): leave it
                for g in by_g[c]:
                    b = inv.get(g)
                    if g == ends.get(a) or (b is not None and cls_p[b] != c) or (b is None and a not in ends):
                        continue
                    improved |= try_move(a, g)
    return {p: ends[p] for p in cls_p if p in ends}


def _edge_keys(d: Diagram, mapping: Optional[dict[str, str]], with_label: bool) -> list[tuple]:
    keys = []
    for e in d.edges:
        s, t = e.source, e.target
        if mapping is not None:
            s, t = mapping.get(s), mapping.get(t)
            if s is None or t is None:
                keys.append(None)  # endpoint not matched -> cannot be correct
                continue
        if not e.directed or e.kind == "bidir":
            s, t = sorted((s, t))
        keys.append((s, t, norm_text(e.label)) if with_label else (s, t))
    return keys


def _count_matches(pred_keys: list, gold_keys: list) -> int:
    pool: dict = {}
    for k in gold_keys:
        pool[k] = pool.get(k, 0) + 1
    hits = 0
    for k in pred_keys:
        if k is not None and pool.get(k, 0) > 0:
            pool[k] -= 1
            hits += 1
    return hits


def score(pred_text: str, gold_text: str) -> dict:
    gold = canonicalize(parse(gold_text)[0])
    pred_raw, errors = parse(pred_text)
    pred = canonicalize(pred_raw)
    mapping, groups = match(pred, gold)
    gold_by_id = {n.id: n for n in gold.nodes}
    matched = [(p, gold_by_id[mapping[p.id]]) for p in pred.nodes if p.id in mapping]
    out = {
        "n": 1,
        "valid": int(not errors),
        "exact": int(dumps(pred) == dumps(gold)),
        "type_ok": int(pred.type == gold.type),
        "pred_nodes": len(pred.nodes), "gold_nodes": len(gold.nodes), "node_hits": len(matched),
        "text_exact": sum(norm_text(p.text) == norm_text(g.text) for p, g in matched),
        "kind_ok": sum(p.kind == g.kind for p, g in matched),
        "pred_edges": len(pred.edges), "gold_edges": len(gold.edges),
    }
    ends = {**groups, **mapping}
    for with_label, name in ((False, "edge_hits"), (True, "edge_label_hits")):
        out[name] = _count_matches(_edge_keys(pred, ends, with_label), _edge_keys(gold, None, with_label))
    return finalize(out)


def finalize(c: dict) -> dict:
    def f1(hits: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
        p = hits / n_pred if n_pred else float(n_gold == 0)
        r = hits / n_gold if n_gold else float(n_pred == 0)
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    c = dict(c)
    c["node_p"], c["node_r"], c["node_f1"] = f1(c["node_hits"], c["pred_nodes"], c["gold_nodes"])
    c["edge_p"], c["edge_r"], c["edge_f1"] = f1(c["edge_hits"], c["pred_edges"], c["gold_edges"])
    c["edge_label_f1"] = f1(c["edge_label_hits"], c["pred_edges"], c["gold_edges"])[2]
    c["text_exact_rate"] = c["text_exact"] / c["node_hits"] if c["node_hits"] else 0.0
    c["kind_acc"] = c["kind_ok"] / c["node_hits"] if c["node_hits"] else 0.0
    return c


COUNT_KEYS = ("n", "valid", "exact", "type_ok", "pred_nodes", "gold_nodes", "node_hits", "text_exact",
              "kind_ok", "pred_edges", "gold_edges", "edge_hits", "edge_label_hits")


def aggregate(results: Iterable[dict]) -> dict:
    """Micro-average: sum the counts, then recompute the ratios."""
    total = dict.fromkeys(COUNT_KEYS, 0)
    for r in results:
        for k in COUNT_KEYS:
            total[k] += r[k]
    out = finalize(total)
    n = max(out["n"], 1)
    out["valid_rate"] = out["valid"] / n
    out["exact_rate"] = out["exact"] / n
    out["type_acc"] = out["type_ok"] / n
    return out
