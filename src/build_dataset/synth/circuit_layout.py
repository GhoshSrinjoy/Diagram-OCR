"""Layered placement and orthogonal wire routing for the synthetic logic circuits.

Canonical frame: signals flow left to right, x grows to the right and y grows DOWN; one unit is
the height of a 2-input gate body. Every node centre and every input pin sits on GRID, so two
horizontal wires are either collinear or at least one grid step apart.

  1. Layers: gates by logic depth (ASAP or ALAP); primary inputs in a column of their own or just
     left of their first consumer; outputs in a last column or right after their driver. A net
     spanning several columns gets one dummy node per column it crosses (Sugiyama layering), so
     wires pass between gates, never through them.
  2. Order inside each layer: barycenter sweeps plus adjacent transpositions (fewest crossings).
     A gate's input pins take their drivers in top-to-bottom order.
  3. y: median-of-neighbours relaxation projected on the order / minimum-separation constraints
     (isotonic regression) and snapped to the grid; then a greedy polish straightens single wires
     and whole long-wire chains, and never leaves the source of one net level with another net's
     sink pin in the same channel (the two wires would read as one line).
  4. x: a column is as wide as its widest gate. In the channel between two columns every net that
     changes height gets its own vertical track; the track order minimises crossings and never
     lets two nets share a line. Junction dots go where three or more wire ends meet.
  5. check_layout() rebuilds connectivity from the drawn geometry alone and rejects anything a
     reader could misread: nets touching or running too close, crossings at corners or dots,
     wires through gates or labels, a net drawn in several pieces, a branch without a dot.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median_low
from typing import Optional

from .circuit_netlist import Netlist
from .circuit_symbols import GRID, SymbolGeom

EPS = 1e-6
W_CONFLICT = 1000  # cost of two nets sharing a line in a channel (i.e. forbidden)
COLLINEAR_GAP = 1.0  # two nets on one line need at least this gap (or a gate in between)

Point = tuple[float, float]
Rect = tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class LayoutParams:
    gate_gap: float = 0.75  # vertical gap between two gate bodies in a column
    wire_gap: float = 0.5  # gap between a gate body and a wire / terminal passing it
    wire_pitch: float = 0.5  # gap between two wires / terminals stacked in a column
    track_pitch: float = 0.35  # spacing of the vertical tracks in a channel
    channel_margin: float = 0.45  # run from a column edge to the nearest track
    align: float = 0.0  # 0: gates left-aligned in their column, 0.5: centred
    layering: str = "asap"  # "asap" | "alap"
    input_mode: str = "edge"  # "edge": inputs in column 0 | "near": just left of their first use
    output_mode: str = "edge"  # "edge": outputs in a last column | "near": right after the driver
    stub_extra: float = 0.0  # near-mode stubs reach this far into their own column
    natural_inputs: bool = True  # keep edge inputs in label order
    vertical_flow: bool = False  # the drawing ends up TB/BT, so labels stack by their width
    label_gap: float = 0.15
    term_r: float = 0.0  # radius of the open terminal circles at stub ends (0 = none)
    min_sep: float = 0.2  # clearance enforced by check_layout


@dataclass
class PlacedGate:
    id: str
    kind: str
    geom: SymbolGeom
    x: float  # input-pin column
    y: float  # centre line
    pin_nets: list[str]  # net on each input pin, top first

    def pins(self) -> list[Point]:
        return [(self.x, self.y + o) for o in self.geom.pin_offsets]

    def out(self) -> Point:
        return (self.x + self.geom.length, self.y)

    def body(self) -> Rect:
        b = self.geom.body
        return (self.x + b[0], self.y + b[1], self.x + b[2], self.y + b[3])


@dataclass
class PlacedTerminal:
    id: str  # i<k> / o<k>
    kind: str  # input | output
    label: str
    net: str
    end: Point  # free end of the stub
    label_rect: Optional[Rect] = None  # where the label ink goes (canonical frame)


@dataclass
class Layout:
    gates: list[PlacedGate]
    terminals: list[PlacedTerminal]
    wires: list[tuple[str, Point, Point]]  # net, p1, p2: merged axis-aligned segments
    dots: list[tuple[str, Point]]
    crossings: int = 0
    stats: dict = field(default_factory=dict)


@dataclass
class _Node:
    id: str
    kind: str  # input | gate | output | dummy
    layer: int
    net: str  # the signal this node drives (inputs, gates, dummies) or receives (outputs)
    half: float = 0.0
    y: int = 0  # grid units


@dataclass
class _ChanNet:
    net: str
    src: str
    dst: list[str]


# --------------------------------------------------------------------------- #
# 1. layers and the layered graph
# --------------------------------------------------------------------------- #
def _levels(nl: Netlist, p: LayoutParams) -> dict[str, int]:
    cons = nl.consumers()
    lev = {f"i{k}": 0 for k in range(len(nl.inputs))}
    for g in nl.gates:
        lev[g.id] = 1 + max(lev[s] for s in g.inputs)
    top = max(lev[g.id] for g in nl.gates)
    out_drivers = {d for _, d in nl.outputs}
    if p.layering == "alap":
        for g in reversed(nl.gates):
            c = [lev[x] for x in cons[g.id]] + ([top + 1] if g.id in out_drivers else [])
            lev[g.id] = min(c) - 1
    for k in range(len(nl.inputs)):
        s = f"i{k}"
        lev[s] = 0 if p.input_mode == "edge" else min(lev[x] for x in cons[s]) - 1
    for j, (_, d) in enumerate(nl.outputs):
        lev[f"o{j}"] = top + 1 if p.output_mode == "edge" else lev[d] + 1
    return lev


def _layer_graph(nl: Netlist, lev: dict[str, int]) -> tuple[dict[str, _Node], list[list[_ChanNet]]]:
    nodes: dict[str, _Node] = {}
    for k in range(len(nl.inputs)):
        nodes[f"i{k}"] = _Node(f"i{k}", "input", lev[f"i{k}"], f"i{k}")
    for g in nl.gates:
        nodes[g.id] = _Node(g.id, "gate", lev[g.id], g.id)
    sinks: dict[str, list[str]] = defaultdict(list)
    for g in nl.gates:
        for s in g.inputs:
            sinks[s].append(g.id)
    for j, (_, d) in enumerate(nl.outputs):
        nodes[f"o{j}"] = _Node(f"o{j}", "output", lev[f"o{j}"], d)
        sinks[d].append(f"o{j}")
    n_layers = max(lev.values()) + 1
    chans: list[list[_ChanNet]] = [[] for _ in range(n_layers - 1)]
    for net in [f"i{k}" for k in range(len(nl.inputs))] + [g.id for g in nl.gates]:
        sk = sinks[net]
        a, far = lev[net], max(lev[x] for x in sk)
        for layer in range(a + 1, far):
            nodes[f"d.{net}.{layer}"] = _Node(f"d.{net}.{layer}", "dummy", layer, net)
        for layer in range(a, far):
            src = net if layer == a else f"d.{net}.{layer}"
            dst = [x for x in sk if lev[x] == layer + 1]
            if layer + 1 < far:
                dst.append(f"d.{net}.{layer + 1}")
            chans[layer].append(_ChanNet(net, src, dst))
    return nodes, chans


# --------------------------------------------------------------------------- #
# 2. order inside layers
# --------------------------------------------------------------------------- #
def _crossings(layers: list[list[str]], succ: dict[str, list[str]], pos: dict[str, int], l0: int) -> int:
    edges = [(pos[u], pos[v]) for u in layers[l0] for v in succ[u]]
    c = 0
    for i in range(len(edges)):
        a = edges[i]
        for j in range(i + 1, len(edges)):
            b = edges[j]
            if (a[0] - b[0]) * (a[1] - b[1]) < 0:
                c += 1
    return c


def _order_layers(layers: list[list[str]], pred: dict[str, list[str]], succ: dict[str, list[str]],
                  fixed: set[int]) -> None:
    n = len(layers)
    pos = {v: i for layer in layers for i, v in enumerate(layer)}

    def total() -> int:
        return sum(_crossings(layers, succ, pos, k) for k in range(n - 1))

    best, best_c = [list(x) for x in layers], total()
    for it in range(12):
        seq = range(1, n) if it % 2 == 0 else range(n - 2, -1, -1)
        nbrs = pred if it % 2 == 0 else succ
        for k in seq:
            if k in fixed:
                continue

            def bary(v: str, nbrs: dict[str, list[str]] = nbrs) -> float:
                nb = nbrs[v]
                return sum(pos[u] for u in nb) / len(nb) if nb else pos[v]

            layers[k].sort(key=lambda v: (bary(v), pos[v]))
            pos.update({v: i for i, v in enumerate(layers[k])})
        c = total()
        if c < best_c:
            best, best_c = [list(x) for x in layers], c
        if best_c == 0:
            break
    layers[:] = best
    pos.update({v: i for layer in layers for i, v in enumerate(layer)})

    def local(k: int) -> int:
        return (_crossings(layers, succ, pos, k - 1) if k > 0 else 0) + \
               (_crossings(layers, succ, pos, k) if k < n - 1 else 0)

    for _ in range(8):  # adjacent transpositions
        improved = False
        for k in range(n):
            if k in fixed:
                continue
            for i in range(len(layers[k]) - 1):
                c0 = local(k)
                a, b = layers[k][i], layers[k][i + 1]
                layers[k][i], layers[k][i + 1] = b, a
                pos[a], pos[b] = i + 1, i
                if local(k) < c0:
                    improved = True
                else:
                    layers[k][i], layers[k][i + 1] = a, b
                    pos[a], pos[b] = i, i + 1
        if not improved:
            break


# --------------------------------------------------------------------------- #
# 3. vertical positions (grid units)
# --------------------------------------------------------------------------- #
def _project(targets: list[float], seps: list[int]) -> list[int]:
    """Closest grid positions (least squares) keeping order and minimum separations."""
    offs = [0]
    for s in seps:
        offs.append(offs[-1] + s)
    blocks: list[list[float]] = []  # pool-adjacent-violators on z = y - offset
    for t, o in zip(targets, offs, strict=True):
        blocks.append([t - o, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    fit: list[float] = []
    for s, c in blocks:
        fit += [s / c] * int(c)
    return [math.floor(z + 0.5) + o for z, o in zip(fit, offs, strict=True)]


def _attach_off(nodes: dict[str, _Node], gates_geom: dict[str, SymbolGeom], pin_nets: dict[str, list[str]],
                v: str, net: str) -> int:
    """Grid offset of the pin of `v` that receives `net` (0 for non-gates)."""
    if nodes[v].kind != "gate":
        return 0
    k = pin_nets[v].index(net)
    return round(gates_geom[v].pin_offsets[k] / GRID)


class _YState:
    """Grid y positions with the cost terms used to polish them."""

    def __init__(self, layers: list[list[str]], nodes: dict[str, _Node], pred: dict[str, list[str]],
                 succ: dict[str, list[str]], seps: list[list[int]], off):
        self.layers, self.nodes, self.pred, self.succ = layers, nodes, pred, succ
        self.seps, self.off = seps, off
        self.slot = {v: (k, i) for k, layer in enumerate(layers) for i, v in enumerate(layer)}

    def wants(self, v: str) -> list[int]:
        n, off = self.nodes, self.off
        return ([n[u].y - off(v, n[u].net) for u in self.pred[v]]
                + [n[w].y + off(w, n[v].net) for w in self.succ[v]])

    def conflicts(self, v: str) -> int:
        """Rows where `v` attaches a wire collinear with another net's wire in the same channel
        (a source of one net level with a sink pin of another): they would read as one line."""
        n, off = self.nodes, self.off
        k = n[v].layer
        c = 0
        if self.succ[v] and k + 1 < len(self.layers):
            for w in self.layers[k + 1]:
                c += sum(n[u].net != n[v].net and n[w].y + off(w, n[u].net) == n[v].y for u in self.pred[w])
        if self.pred[v] and k > 0:
            for u in self.pred[v]:
                yy = n[v].y + off(v, n[u].net)
                c += sum(bool(self.succ[x]) and n[x].net != n[u].net and n[x].y == yy
                         for x in self.layers[k - 1])
        return c

    def cost(self, v: str) -> float:
        y = self.nodes[v].y
        return sum((y != t) + 0.02 * abs(y - t) for t in self.wants(v)) + 50 * self.conflicts(v)

    def bounds(self, v: str) -> tuple[int, int]:
        k, i = self.slot[v]
        layer = self.layers[k]
        lo = self.nodes[layer[i - 1]].y + self.seps[k][i - 1] if i > 0 else -10**9
        hi = self.nodes[layer[i + 1]].y - self.seps[k][i] if i + 1 < len(layer) else 10**9
        return lo, hi

    def total_conflicts(self) -> int:
        return sum(self.conflicts(v) for layer in self.layers for v in layer if self.succ[v])


def _straighten(st: _YState, chains: list[list[str]]) -> None:
    """Greedy polish: single-node moves inside their free interval, and whole long-wire chains
    (dummies, plus a primary input at their head) moved to one common row."""
    nodes = st.nodes
    for _ in range(10):
        changed = False
        for layer in st.layers:
            for v in layer:
                lo, hi = st.bounds(v)
                cur = nodes[v].y
                best, best_c = cur, st.cost(v)
                for t in sorted(set(st.wants(v)) | {cur - 2, cur - 1, cur + 1, cur + 2}):
                    if t == cur or not lo <= t <= hi:
                        continue
                    nodes[v].y = t
                    c = st.cost(v)
                    if c < best_c - 1e-9:
                        best, best_c = t, c
                nodes[v].y = best
                changed |= best != cur
        for chain in chains:
            cur = [nodes[v].y for v in chain]
            near = set(chain) | {w for v in chain for w in st.succ[v]}
            near |= {u for v in chain for u in st.pred[v]}

            best, best_c = cur, sum(st.cost(x) for x in near)
            for t in sorted({y for v in chain for y in st.wants(v)} | set(cur)):
                if not all(lo <= t <= hi for lo, hi in map(st.bounds, chain)):
                    continue
                for v in chain:
                    nodes[v].y = t
                c = sum(st.cost(x) for x in near)
                if c < best_c - 1e-9:
                    best, best_c = [t] * len(chain), c
                for v, y in zip(chain, cur, strict=True):
                    nodes[v].y = y
            for v, y in zip(chain, best, strict=True):
                nodes[v].y = y
            changed |= best != cur
        if not changed:
            break


# --------------------------------------------------------------------------- #
# 4. channel routing
# --------------------------------------------------------------------------- #
@dataclass
class _Track:
    net: str
    ys: int  # source attach
    sinks: list[int]
    attach: frozenset
    lo: int
    hi: int


def _track_cost(order: list[_Track], straight: list[int]) -> int:
    cost = 0
    n = len(order)
    for ia, a in enumerate(order):
        hs = [(a.ys, 0, ia)] + [(y, ia + 1, n) for y in a.sinks]
        for y, b0, b1 in hs:
            for ib in range(b0, b1):
                b = order[ib]
                if y in b.attach:
                    cost += W_CONFLICT
                elif b.lo < y < b.hi:
                    cost += 1
    for y in straight:
        for b in order:
            if y in b.attach:
                cost += W_CONFLICT
            elif b.lo < y < b.hi:
                cost += 1
    return cost


def _order_tracks(tracks: list[_Track], straight: list[int]) -> tuple[list[_Track], int]:
    if not tracks:
        return [], _track_cost([], straight)
    starts = [
        sorted(tracks, key=lambda t: (t.ys, t.lo)),
        sorted(tracks, key=lambda t: (sum(t.sinks) / len(t.sinks), t.ys)),
        sorted(tracks, key=lambda t: (t.ys > min(t.sinks), t.ys if t.ys > min(t.sinks) else -t.ys)),
    ]
    best, best_c = None, None
    for order in starts:
        order = list(order)
        c = _track_cost(order, straight)
        for _ in range(6):  # first-improvement insertion moves
            improved = False
            for i in range(len(order)):
                for j in range(len(order)):
                    if i == j:
                        continue
                    cand = list(order)
                    cand.insert(j, cand.pop(i))
                    cc = _track_cost(cand, straight)
                    if cc < c:
                        order, c, improved = cand, cc, True
            if not improved:
                break
        if best_c is None or c < best_c:
            best, best_c = order, c
        if best_c == 0:
            break
    assert best is not None and best_c is not None
    return best, best_c


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def layout_circuit(nl: Netlist, geoms: dict[str, SymbolGeom], label_size: dict[str, tuple[float, float]],
                   p: LayoutParams) -> Optional[Layout]:
    """Place and route `nl`. geoms: gate id -> symbol; label_size: terminal id -> (w, h) of its
    label ink (empty for bare stubs). None when no clean routing was found."""
    lev = _levels(nl, p)
    nodes, chans = _layer_graph(nl, lev)
    n_layers = len(chans) + 1
    pred: dict[str, list[str]] = defaultdict(list)
    succ: dict[str, list[str]] = defaultdict(list)
    for chan in chans:
        for cn in chan:
            for d in cn.dst:
                succ[cn.src].append(d)
                pred[d].append(cn.src)

    # initial order: inputs by label, then each layer by its first predecessor
    layers: list[list[str]] = [[] for _ in range(n_layers)]
    for v in nodes.values():
        layers[v.layer].append(v.id)
    rank = {f"i{k}": k for k in range(len(nl.inputs))}
    rank.update({g.id: 100 + j for j, g in enumerate(nl.gates)})
    rank.update({f"o{j}": 1000 + j for j in range(len(nl.outputs))})
    pos: dict[str, float] = {}
    for k in range(n_layers):
        layers[k].sort(key=lambda v: (min((pos[u] for u in pred[v]), default=0.0),
                                      rank.get(v, rank.get(nodes[v].net, 0) + 0.5)))
        pos.update({v: i for i, v in enumerate(layers[k])})
    fixed = {0} if p.input_mode == "edge" and p.natural_inputs else set()
    _order_layers(layers, pred, succ, fixed)
    pos = {v: i for layer in layers for i, v in enumerate(layer)}

    # pins take their drivers top to bottom
    pin_nets = {g.id: [nodes[u].net for u in sorted(pred[g.id], key=lambda u: pos[u])] for g in nl.gates}
    gates_geom = {g.id: geoms[g.id] for g in nl.gates}

    # node extents along the stacking axis
    for v in nodes.values():
        if v.kind == "gate":
            v.half = gates_geom[v.id].half_h
        elif v.kind in ("input", "output") and v.id in label_size:
            w, h = label_size[v.id]
            v.half = max((w if p.vertical_flow else h) / 2, p.term_r)
        elif v.kind in ("input", "output"):
            v.half = p.term_r

    def gap(a: _Node, b: _Node) -> float:
        ga, gb = a.kind == "gate", b.kind == "gate"
        return p.gate_gap if ga and gb else (p.wire_gap if ga or gb else p.wire_pitch)

    seps = [[max(1, math.ceil((nodes[a].half + nodes[b].half + gap(nodes[a], nodes[b])) / GRID - 1e-9))
             for a, b in zip(layer, layer[1:], strict=False)] for layer in layers]
    for k, layer in enumerate(layers):
        ys = _project([0.0] * len(layer), seps[k])
        mid = (ys[0] + ys[-1]) // 2
        for v, y in zip(layer, ys, strict=True):
            nodes[v].y = y - mid

    def off(v: str, net: str) -> int:
        return _attach_off(nodes, gates_geom, pin_nets, v, net)

    for it in range(16):
        seq = range(n_layers) if it % 2 == 0 else range(n_layers - 1, -1, -1)
        for k in seq:
            targets = []
            for v in layers[k]:
                vals = [nodes[u].y - off(v, nodes[u].net) for u in pred[v]]
                vals += [nodes[w].y + off(w, nodes[v].net) for w in succ[v]]
                targets.append(float(median_low(vals)) if vals else float(nodes[v].y))
            for v, y in zip(layers[k], _project(targets, seps[k]), strict=True):
                nodes[v].y = y
    yst = _YState(layers, nodes, pred, succ, seps, off)
    chains: dict[str, list[str]] = defaultdict(list)
    for v in nodes.values():
        if v.kind == "dummy":
            chains[v.net].append(v.id)
    chain_list = []
    for net, ds in chains.items():
        ds.sort(key=lambda x: nodes[x].layer)
        chain_list.append(([net] if nodes[net].kind == "input" else []) + ds)
    _straighten(yst, chain_list)
    if yst.total_conflicts():
        return None

    # ---- channels: tracks
    widths = [max((gates_geom[v].length for v in layer if nodes[v].kind == "gate"), default=0.0)
              for layer in layers]
    chan_tracks: list[list[_Track]] = []
    n_cross = 0
    for chan in chans:
        tracks, straight = [], []
        for cn in chan:
            ys = nodes[cn.src].y
            sinks = [nodes[d].y + off(d, cn.net) for d in cn.dst]
            att = frozenset([ys, *sinks])
            if len(att) == 1:
                straight.append(ys)
            else:
                tracks.append(_Track(cn.net, ys, sinks, att, min(att), max(att)))
        order, cost = _order_tracks(tracks, straight)
        if cost >= W_CONFLICT:
            return None
        n_cross += cost
        chan_tracks.append(order)

    # ---- x positions
    x_left = [0.0] * n_layers
    x_right = [0.0] * n_layers
    track_x: list[dict[str, float]] = []
    for k in range(n_layers):
        x_right[k] = x_left[k] + widths[k]
        if k < n_layers - 1:
            t = chan_tracks[k]
            x0 = x_right[k] + p.channel_margin
            track_x.append({tr.net: x0 + i * p.track_pitch for i, tr in enumerate(t)})
            span = 2 * p.channel_margin + max(0, len(t) - 1) * p.track_pitch
            x_left[k + 1] = x_right[k] + span

    gx = {g.id: x_left[lev[g.id]] + p.align * (widths[lev[g.id]] - gates_geom[g.id].length) for g in nl.gates}
    placed = [PlacedGate(g.id, g.kind, gates_geom[g.id], gx[g.id], nodes[g.id].y * GRID, pin_nets[g.id])
              for g in nl.gates]
    by_gate = {pg.id: pg for pg in placed}

    def src_x(u: str) -> float:
        return by_gate[u].out()[0] if nodes[u].kind == "gate" else x_right[nodes[u].layer]

    def dst_x(v: str) -> float:
        return by_gate[v].x if nodes[v].kind == "gate" else x_left[nodes[v].layer]

    segs: list[tuple[str, float, float, float, float]] = []
    dots: list[tuple[str, Point]] = []
    for k, chan in enumerate(chans):
        for cn in chan:
            ys = nodes[cn.src].y
            sinks = [(nodes[d].y + off(d, cn.net), dst_x(d)) for d in cn.dst]
            x0 = src_x(cn.src)
            if cn.net not in track_x[k]:  # straight through the channel
                segs.append((cn.net, x0, ys * GRID, sinks[0][1], ys * GRID))
                continue
            xt = track_x[k][cn.net]
            lo, hi = min([ys] + [s[0] for s in sinks]), max([ys] + [s[0] for s in sinks])
            segs.append((cn.net, x0, ys * GRID, xt, ys * GRID))
            segs.append((cn.net, xt, lo * GRID, xt, hi * GRID))
            for y, x1 in sinks:
                segs.append((cn.net, xt, y * GRID, x1, y * GRID))
            sink_ys = {s[0] for s in sinks}
            for y in sorted({ys} | sink_ys):
                deg = (y == ys) + (y in sink_ys) + (y > lo) + (y < hi)
                if deg >= 3:
                    dots.append((cn.net, (xt, y * GRID)))
    for v in nodes.values():  # dummies cross their column
        if v.kind == "dummy" and widths[v.layer] > 0:
            segs.append((v.net, x_left[v.layer], v.y * GRID, x_right[v.layer], v.y * GRID))

    terminals: list[PlacedTerminal] = []
    labels = {f"i{k}": s for k, s in enumerate(nl.inputs)}
    labels.update({f"o{j}": s for j, (s, _) in enumerate(nl.outputs)})
    for v in nodes.values():
        if v.kind not in ("input", "output"):
            continue
        y = v.y * GRID
        w_col = widths[v.layer]
        if v.kind == "input":
            x_end = x_right[v.layer] - (min(w_col, p.stub_extra) if p.input_mode == "near" else 0.0)
            if x_end < x_right[v.layer] - EPS:
                segs.append((v.net, x_end, y, x_right[v.layer], y))
        else:
            x_end = x_left[v.layer] + (min(w_col, p.stub_extra) if p.output_mode == "near" else 0.0)
            if x_end > x_left[v.layer] + EPS:
                segs.append((v.net, x_left[v.layer], y, x_end, y))
        rect = None
        if v.id in label_size:
            w, h = label_size[v.id]
            ext_flow, ext_stack = (h, w) if p.vertical_flow else (w, h)
            d0 = p.term_r + p.label_gap
            if v.kind == "input":
                rect = (x_end - d0 - ext_flow, y - ext_stack / 2, x_end - d0, y + ext_stack / 2)
            else:
                rect = (x_end + d0, y - ext_stack / 2, x_end + d0 + ext_flow, y + ext_stack / 2)
        terminals.append(PlacedTerminal(v.id, v.kind, labels[v.id], v.net, (x_end, y), rect))

    wires = [(net, (x1, y1), (x2, y2)) for net, x1, y1, x2, y2 in _merge(segs)]
    lay = Layout(placed, sorted(terminals, key=lambda t: t.id), wires, dots, n_cross,
                 stats={"layers": n_layers, "tracks": sum(len(t) for t in chan_tracks)})
    return lay


_RawSeg = tuple[str, float, float, float, float]  # net, x1, y1, x2, y2


def _merge(segs: list[_RawSeg]) -> list[_RawSeg]:
    """Merge collinear touching segments of the same net; drop zero-length ones."""
    groups: dict[tuple, list[list[float]]] = defaultdict(list)
    for net, x1, y1, x2, y2 in segs:
        if abs(x1 - x2) < EPS and abs(y1 - y2) < EPS:
            continue
        if abs(y1 - y2) < EPS:
            groups[(net, "h", round(y1, 6))].append([min(x1, x2), max(x1, x2)])
        else:
            assert abs(x1 - x2) < EPS, "wires are axis-aligned"
            groups[(net, "v", round(x1, 6))].append([min(y1, y2), max(y1, y2)])
    out = []
    for (net, o, c), ivs in groups.items():
        ivs.sort()
        cur = ivs[0]
        merged = []
        for a, b in ivs[1:]:
            if a <= cur[1] + EPS:
                cur[1] = max(cur[1], b)
            else:
                merged.append(cur)
                cur = [a, b]
        merged.append(cur)
        for a, b in merged:
            out.append((net, a, c, b, c) if o == "h" else (net, c, a, c, b))
    return sorted(out)


# --------------------------------------------------------------------------- #
# 5. validation from geometry
# --------------------------------------------------------------------------- #
_Seg = tuple[str, float, float, float, float, Optional[str]]  # net, x1, y1, x2, y2, owning gate (leads)


def _segments_with_leads(lay: Layout) -> list[_Seg]:
    """Wires plus gate leads (pin -> body), normalised to x1<=x2, y1<=y2."""
    out: list[_Seg] = [(net, min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]), None)
                       for net, a, b in lay.wires]
    for g in lay.gates:
        x0, _, x1, _ = g.body()
        for net, (px, py) in zip(g.pin_nets, g.pins(), strict=True):
            out.append((net, px, py, max(px, x0), py, g.id))
        ox, oy = g.out()
        out.append((g.id, min(x1, ox), oy, ox, oy, g.id))
    return out


def _gate_between(lay: Layout, horizontal: bool, c: float, a: float, b: float) -> bool:
    """Is the gap (a, b) on line `c` bridged by a gate body (e.g. the in / out leads of a NOT)?"""
    m = (a + b) / 2
    for g in lay.gates:
        x0, y0, x1, y1 = g.body()
        if horizontal and y0 < c < y1 and x0 <= m <= x1:
            return True
        if not horizontal and x0 < c < x1 and y0 <= m <= y1:
            return True
    return False


def _rect_hits(r: Rect, x1: float, y1: float, x2: float, y2: float, pad: float) -> bool:
    return x1 < r[2] + pad and x2 > r[0] - pad and y1 < r[3] + pad and y2 > r[1] - pad


def check_layout(lay: Layout, min_sep: float = 0.2) -> list[str]:
    """Everything a reader could trip over; an empty list means the drawing is unambiguous.

    Connectivity is rebuilt from the geometry alone: segments of one net must form one piece
    (wires end-to-end, L corners, and T / 4-way junctions that carry a dot), while segments of
    different nets may only cross cleanly - away from ends, corners and dots."""
    ms = min_sep
    probs: list[str] = []
    segs = _segments_with_leads(lay)
    n = len(segs)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dot_pts = {(round(x, 6), round(y, 6)) for _, (x, y) in lay.dots}
    junctions: set[tuple[float, float]] = set()
    for i in range(n):
        ni, ax1, ay1, ax2, ay2, _ = segs[i]
        ah = abs(ay1 - ay2) < EPS
        for j in range(i + 1, n):
            nj, bx1, by1, bx2, by2, _ = segs[j]
            bh = abs(by1 - by2) < EPS
            same = ni == nj
            if ah == bh:  # parallel
                if ah:
                    d, o1, o2 = abs(ay1 - by1), (ax1, ax2), (bx1, bx2)
                else:
                    d, o1, o2 = abs(ax1 - bx1), (ay1, ay2), (by1, by2)
                gap = max(o1[0], o2[0]) - min(o1[1], o2[1])  # < 0: the extents overlap
                if d < EPS:  # collinear
                    if same and gap <= EPS:
                        parent[find(i)] = find(j)  # continuation (wire into a lead)
                    elif gap < (ms if same else COLLINEAR_GAP) and not _gate_between(
                            lay, ah, ay1 if ah else ax1, min(o1[1], o2[1]), max(o1[0], o2[0])):
                        probs.append(f"{ni}/{nj}: collinear wires read as one")
                elif d < ms and gap < ms:
                    probs.append(f"{ni}/{nj}: parallel wires too close")
                continue
            h, v = (segs[i], segs[j]) if ah else (segs[j], segs[i])
            hy, hx1, hx2 = h[2], h[1], h[3]
            vx, vy1, vy2 = v[1], v[2], v[4]
            if not (hx1 - ms < vx < hx2 + ms and vy1 - ms < hy < vy2 + ms):
                continue
            pt = (round(vx, 6), round(hy, 6))
            inside = hx1 + ms <= vx <= hx2 - ms and vy1 + ms <= hy <= vy2 - ms
            if not same:
                if not inside:
                    probs.append(f"{ni}/{nj}: nets touch or nearly touch")
                elif pt in dot_pts:
                    probs.append(f"{ni}/{nj}: crossing on a dot")
                continue
            touching = hx1 - EPS <= vx <= hx2 + EPS and vy1 - EPS <= hy <= vy2 + EPS
            if not touching:
                probs.append(f"{ni}: near miss with itself")
                continue
            h_end = abs(vx - hx1) < EPS or abs(vx - hx2) < EPS
            v_end = abs(hy - vy1) < EPS or abs(hy - vy2) < EPS
            if h_end and v_end:  # L corner
                parent[find(i)] = find(j)
            elif (h_end or v_end or inside) and pt in dot_pts:  # T or 4-way junction
                parent[find(i)] = find(j)
                junctions.add(pt)
            else:
                probs.append(f"{ni}: branch without a dot / self crossing at {pt}")
    for pt in dot_pts - junctions:
        probs.append(f"stray dot at {pt}")

    pieces: dict[str, set[int]] = defaultdict(set)
    for i, s in enumerate(segs):
        pieces[s[0]].add(find(i))
    for net, comps in pieces.items():
        if len(comps) > 1:
            probs.append(f"{net}: drawn in {len(comps)} pieces")

    # obstacles: gate bodies and labels
    bodies = [(g.id, g.body()) for g in lay.gates]
    labels = [t.label_rect for t in lay.terminals if t.label_rect]
    pin_rows = {g.id: {(round(px, 6), round(py, 6)) for px, py in g.pins() + [g.out()]} for g in lay.gates}
    for net, x1, y1, x2, y2, owner in segs:
        for gid, r in bodies:
            if gid == owner:
                continue
            if abs(y1 - y2) < EPS and ({(round(x1, 6), round(y1, 6)), (round(x2, 6), round(y2, 6))}
                                       & pin_rows[gid]):
                # the wire into / out of this gate's own pin: it may come close to the body
                if not _rect_hits(r, x1, y1, x2, y2, -EPS):
                    continue
            if _rect_hits(r, x1, y1, x2, y2, ms / 2):
                probs.append(f"{net}: wire through gate {gid}")
        for r in labels:
            if _rect_hits(r, x1, y1, x2, y2, ms / 2):
                probs.append(f"{net}: wire through a label")
    rects = [r for _, r in bodies] + labels
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            if a[0] < b[2] + ms and b[0] < a[2] + ms and a[1] < b[3] + ms and b[1] < a[3] + ms:
                probs.append("labels / gates overlap")
    return probs
