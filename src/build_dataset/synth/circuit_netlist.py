"""Combinational netlists for the synthetic logic-circuit generator (source synth_circuits).

A netlist is a DAG of gates over named primary inputs; outputs are named taps on gate signals.
Signals are referenced by id: primary input k is "i{k}", gate k is "g{k}" (gates are stored in
topological order). Every gate has distinct drivers on its pins, every primary input is used and
every gate lies on a path to an output - so a drawing never has dangling parts.

random_netlist() draws random DAGs; classic_netlist() returns one of a dozen textbook circuits
(adders, multiplexer, decoder, comparator, ...) with conventional pin names.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

GATE_KINDS = ("and", "or", "not", "nand", "nor", "xor", "xnor", "buf")
ONE_INPUT = frozenset({"not", "buf"})
THREE_INPUT_OK = frozenset({"and", "or", "nand", "nor"})
MAX_GATES = 16
MAX_RANDOM_OUTPUTS = 3

# Roughly the LogicBench-1K gate shares, with the rare kinds nudged up so each is well covered.
KIND_WEIGHTS = {"and": 0.21, "not": 0.17, "or": 0.15, "nand": 0.14, "nor": 0.11, "xor": 0.11,
                "xnor": 0.07, "buf": 0.04}
# Gates per random circuit (LogicBench-1K: median 5, mean 6.2).
N_GATES_WEIGHTS = {2: 5, 3: 9, 4: 12, 5: 13, 6: 12, 7: 11, 8: 9, 9: 8, 10: 6, 11: 5, 12: 4, 13: 3,
                   14: 2, 15: 1.5, 16: 1}

INPUT_NAME_SETS = (
    ["A", "B", "C", "D", "E", "F"], ["a", "b", "c", "d", "e", "f"],
    ["x1", "x2", "x3", "x4", "x5", "x6"], ["x0", "x1", "x2", "x3", "x4", "x5"],
    ["I0", "I1", "I2", "I3", "I4", "I5"], ["P", "Q", "R", "S", "T", "U"],
    ["in1", "in2", "in3", "in4", "in5", "in6"], ["A1", "A2", "A3", "A4", "A5", "A6"],
    ["X", "Y", "Z", "W", "V", "U"],
)
OUTPUT_NAME_SETS = (
    ["Y", "Z", "W"], ["F", "G", "H"], ["Y1", "Y2", "Y3"], ["F1", "F2", "F3"], ["Q", "R", "S"],
    ["out1", "out2", "out3"], ["Z1", "Z2", "Z3"], ["O1", "O2", "O3"], ["X", "Y", "Z"],
)
SINGLE_OUTPUT_NAMES = ("Y", "F", "Z", "Q", "X", "Out", "out", "Y0", "F1", "G")


@dataclass
class Gate:
    id: str
    kind: str
    inputs: list[str]  # driver signal per pin


@dataclass
class Netlist:
    inputs: list[str]  # labels; the signal id of inputs[k] is f"i{k}"
    gates: list[Gate]  # topological order
    outputs: list[tuple[str, str]]  # (label, driver signal id)
    classic: Optional[str] = None

    def consumers(self) -> dict[str, list[str]]:
        """signal id -> ids of the gates it feeds (one entry per pin)."""
        out: dict[str, list[str]] = {f"i{k}": [] for k in range(len(self.inputs))}
        out.update({g.id: [] for g in self.gates})
        for g in self.gates:
            for s in g.inputs:
                out[s].append(g.id)
        return out

    def check(self) -> None:
        """Assert the invariants the layout and the targets rely on."""
        known = {f"i{k}" for k in range(len(self.inputs))}
        for g in self.gates:
            assert g.kind in GATE_KINDS, g.kind
            assert len(g.inputs) in ((1,) if g.kind in ONE_INPUT else (2, 3)), f"{g.id}: fan-in"
            assert len(set(g.inputs)) == len(g.inputs), f"{g.id}: repeated driver"
            assert all(s in known for s in g.inputs), f"{g.id}: not topological"
            known.add(g.id)
        cons = self.consumers()
        driven = {d for _, d in self.outputs}
        assert all(d in known for d in driven)
        for s, users in cons.items():
            assert users or s in driven, f"{s} drives nothing"
        labels = list(self.inputs) + [lbl for lbl, _ in self.outputs]
        assert len(set(labels)) == len(labels), "duplicate labels"


def _weighted(rng: random.Random, weights: dict) -> object:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _levels(gates: list[Gate]) -> dict[str, int]:
    lev: dict[str, int] = {}
    for g in gates:
        lev[g.id] = 1 + max(lev.get(s, 0) for s in g.inputs)
    return lev


# --------------------------------------------------------------------------- #
# Random DAGs
# --------------------------------------------------------------------------- #
def random_netlist(rng: random.Random, n_gates: Optional[int] = None) -> Optional[Netlist]:
    """A random circuit with 2-6 inputs, 2-16 gates and 1-3 outputs (None: retry with another rng).

    Gates are drawn level by level: a gate at level L has one driver from level L-1 and the others
    from any lower level, so the logic depth is controlled (LogicBench-1K: depth ~ 1.6 + 0.27 * gates).
    """
    n_gates = n_gates or int(_weighted(rng, N_GATES_WEIGHTS))
    depth = max(1, min(n_gates, round(rng.gauss(1.6 + 0.27 * n_gates, 0.7))))
    n_in = max(2, min(6, round(rng.gauss(1.4 + 0.33 * n_gates, 0.9))))
    n_out = min(int(_weighted(rng, {1: 0.5, 2: 0.33, 3: 0.17})), n_gates)
    per_level = [1] * depth
    for _ in range(n_gates - depth):
        per_level[rng.randrange(depth)] += 1

    by_level: list[list[str]] = [[f"i{k}" for k in range(n_in)]]
    fanout = {f"i{k}": 0 for k in range(n_in)}
    kind_of: dict[str, str] = {}
    gates: list[Gate] = []
    seen: set[tuple[str, frozenset]] = set()
    for level in range(1, depth + 1):
        by_level.append([])
        for _ in range(per_level[level - 1]):
            gid = f"g{len(gates)}"
            for _attempt in range(16):
                kind = str(_weighted(rng, KIND_WEIGHTS))
                k = 1 if kind in ONE_INPUT else (3 if kind in THREE_INPUT_OK and rng.random() < 0.13 else 2)
                chosen = _pick_drivers(rng, k, kind, by_level, fanout, kind_of)
                if chosen is None or (kind, frozenset(chosen)) in seen:
                    continue
                seen.add((kind, frozenset(chosen)))
                break
            else:
                return None
            gates.append(Gate(gid, kind, chosen))
            for sig in chosen:
                fanout[sig] += 1
            kind_of[gid] = kind
            by_level[level].append(gid)
            fanout[gid] = 0

    if not _use_all_inputs(rng, gates, n_in, fanout):
        return None

    # Dangling gates become outputs; merge them while there are too many.
    dangling = [g.id for g in gates if fanout[g.id] == 0]
    while len(dangling) > MAX_RANDOM_OUTPUTS:
        if len(gates) >= MAX_GATES:
            return None
        take = 3 if len(dangling) > MAX_RANDOM_OUTPUTS + 1 and rng.random() < 0.3 else 2
        pick = dangling[:take]
        kind = rng.choice(["and", "or", "xor", "nand", "nor"] if take == 2 else ["and", "or", "nand", "nor"])
        gid = f"g{len(gates)}"
        gates.append(Gate(gid, kind, pick))
        for sig in pick:
            fanout[sig] += 1
        fanout[gid] = 0
        dangling = dangling[take:] + [gid]
    outputs = list(dangling)
    # Extra taps on internal gates (deepest first) until the output count is reached.
    lev = _levels(gates)
    internal = sorted((g.id for g in gates if g.id not in outputs), key=lambda sig: (-lev[sig], rng.random()))
    while len(outputs) < n_out and internal:
        outputs.append(internal.pop(0))
    outputs.sort(key=lambda sig: int(sig[1:]))

    in_names, out_names = _names(rng, n_in, len(outputs))
    nl = Netlist(in_names, gates, list(zip(out_names, outputs, strict=True)))
    nl.check()
    return nl


def _pick_drivers(rng: random.Random, k: int, kind: str, by_level: list[list[str]], fanout: dict[str, int],
                  kind_of: dict[str, str]) -> Optional[list[str]]:
    """k distinct drivers for a gate at level len(by_level) - 1: one from the level just below."""
    prev = by_level[-2]
    lower = [sig for lv in by_level[:-1] for sig in lv]
    if k > len(lower):
        return None
    unused_prev = [sig for sig in prev if fanout[sig] == 0]
    first = rng.choice(unused_prev) if unused_prev and rng.random() < 0.8 else rng.choice(prev)
    chosen = [first]
    while len(chosen) < k:
        pool = [sig for sig in lower if sig not in chosen]
        unused_in = [sig for sig in pool if sig[0] == "i" and fanout[sig] == 0]
        unused_g = [sig for sig in pool if sig[0] == "g" and fanout[sig] == 0]
        if unused_in and rng.random() < 0.7:
            chosen.append(rng.choice(unused_in))
        elif unused_g and rng.random() < 0.75:
            chosen.append(rng.choice(unused_g))
        else:
            chosen.append(rng.choice(pool))
    if kind in ONE_INPUT and kind_of.get(chosen[0]) in ONE_INPUT and rng.random() < 0.9:
        return None  # NOT -> NOT chains look like noise
    rng.shuffle(chosen)
    return chosen


def _use_all_inputs(rng: random.Random, gates: list[Gate], n_in: int, fanout: dict[str, int]) -> bool:
    """Rewire pins so that every primary input drives something (keeps the DAG acyclic)."""
    for k in range(n_in):
        u = f"i{k}"
        if fanout[u]:
            continue
        slots = [(g, j) for g in gates for j, s in enumerate(g.inputs)
                 if fanout[s] >= 2 and u not in g.inputs]
        if not slots:
            return False
        g, j = rng.choice(slots)
        fanout[g.inputs[j]] -= 1
        g.inputs[j] = u
        fanout[u] += 1
    return True


def _names(rng: random.Random, n_in: int, n_out: int) -> tuple[list[str], list[str]]:
    ins = rng.choice(INPUT_NAME_SETS)[:n_in]
    for _ in range(20):
        if n_out == 1:
            outs = [rng.choice(SINGLE_OUTPUT_NAMES)]
        else:
            outs = rng.choice(OUTPUT_NAME_SETS)[:n_out]
        if not set(outs) & set(ins):
            return list(ins), list(outs)
    return list(ins), [f"out{j + 1}" for j in range(n_out)]


# --------------------------------------------------------------------------- #
# Textbook circuits
# --------------------------------------------------------------------------- #
def _nl(name: str, inputs: list[str], spec: list[tuple[str, list[str]]],
        outs: list[tuple[str, str]]) -> Netlist:
    """spec: (kind, drivers) per gate in order, drivers given as input labels or g<k>."""
    ref = {lbl: f"i{k}" for k, lbl in enumerate(inputs)}
    gates = [Gate(f"g{j}", kind, [ref.get(s, s) for s in drv]) for j, (kind, drv) in enumerate(spec)]
    return Netlist(list(inputs), gates, list(outs), classic=name)


def _half_adder(rng: random.Random) -> Netlist:
    a, b = rng.choice([("A", "B"), ("X", "Y"), ("a", "b"), ("x", "y"), ("A0", "B0")])
    s, c = rng.choice([("S", "C"), ("Sum", "Carry"), ("S", "Cout"), ("S0", "C1")])
    return _nl("half_adder", [a, b], [("xor", [a, b]), ("and", [a, b])], [(s, "g0"), (c, "g1")])


def _full_adder(rng: random.Random) -> Netlist:
    a, b, ci = rng.choice([("A", "B", "Cin"), ("A", "B", "Ci"), ("X", "Y", "Z"), ("a", "b", "cin"),
                           ("A", "B", "C")])
    s, co = rng.choice([("S", "Cout"), ("Sum", "Cout"), ("S", "Co"), ("S", "C")] if ci != "C"
                       else [("S", "Cout"), ("Sum", "Carry")])
    spec = [("xor", [a, b]), ("xor", ["g0", ci]), ("and", [a, b]), ("and", ["g0", ci]), ("or", ["g3", "g2"])]
    return _nl("full_adder", [a, b, ci], spec, [(s, "g1"), (co, "g4")])


def _mux2(rng: random.Random) -> Netlist:
    i0, i1, sel = rng.choice([("I0", "I1", "S"), ("D0", "D1", "S"), ("A", "B", "S"), ("a", "b", "s"),
                              ("I0", "I1", "Sel")])
    y = rng.choice(["Y", "F", "Out", "Z"])
    spec = [("not", [sel]), ("and", [i0, "g0"]), ("and", [i1, sel]), ("or", ["g1", "g2"])]
    return _nl("mux2", [i0, i1, sel], spec, [(y, "g3")])


def _demux2(rng: random.Random) -> Netlist:
    d, sel = rng.choice([("D", "S"), ("I", "S"), ("D", "Sel"), ("x", "s")])
    y0, y1 = rng.choice([("Y0", "Y1"), ("O0", "O1"), ("D0", "D1") if d != "D" else ("Y0", "Y1")])
    spec = [("not", [sel]), ("and", [d, "g0"]), ("and", [d, sel])]
    return _nl("demux2", [d, sel], spec, [(y0, "g1"), (y1, "g2")])


def _majority(rng: random.Random) -> Netlist:
    a, b, c = rng.choice([("A", "B", "C"), ("x", "y", "z"), ("x1", "x2", "x3")])
    y = rng.choice(["Y", "M", "F", "Maj"])
    if rng.random() < 0.6:
        spec = [("and", [a, b]), ("and", [b, c]), ("and", [a, c]), ("or", ["g0", "g1", "g2"])]
        return _nl("majority", [a, b, c], spec, [(y, "g3")])
    spec = [("and", [a, b]), ("and", [b, c]), ("and", [a, c]), ("or", ["g0", "g1"]), ("or", ["g3", "g2"])]
    return _nl("majority", [a, b, c], spec, [(y, "g4")])


def _parity(rng: random.Random) -> Netlist:
    n = rng.choice([3, 4, 4, 5])
    names = rng.choice([["A", "B", "C", "D", "E"], ["x1", "x2", "x3", "x4", "x5"],
                        ["D0", "D1", "D2", "D3", "D4"], ["a", "b", "c", "d", "e"]])[:n]
    last = "xnor" if rng.random() < 0.25 else "xor"
    spec: list[tuple[str, list[str]]] = []
    if rng.random() < 0.5:  # balanced tree
        layer = list(names)
        while len(layer) > 1:
            nxt = []
            for j in range(0, len(layer) - 1, 2):
                spec.append(("xor", [layer[j], layer[j + 1]]))
                nxt.append(f"g{len(spec) - 1}")
            if len(layer) % 2:
                nxt.append(layer[-1])
            layer = nxt
    else:  # chain
        cur = names[0]
        for s in names[1:]:
            spec.append(("xor", [cur, s]))
            cur = f"g{len(spec) - 1}"
    spec[-1] = (last, spec[-1][1])
    p = rng.choice([s for s in (["P", "Y", "Par", "E"] if last == "xnor" else ["P", "Y", "Par", "O"])
                    if s not in names])
    return _nl("parity", names, spec, [(p, f"g{len(spec) - 1}")])


def _decoder2to4(rng: random.Random) -> Netlist:
    a, b = rng.choice([("A", "B"), ("A1", "A0"), ("S1", "S0"), ("x", "y")])
    outs = rng.choice([["D0", "D1", "D2", "D3"], ["Y0", "Y1", "Y2", "Y3"], ["O0", "O1", "O2", "O3"]])
    if rng.random() < 0.3:  # with enable: 3-input ANDs
        en = "EN" if rng.random() < 0.5 else "E"
        spec = [("not", [a]), ("not", [b]), ("and", ["g0", "g1", en]), ("and", ["g0", b, en]),
                ("and", [a, "g1", en]), ("and", [a, b, en])]
        ins = [a, b, en]
    else:
        spec = [("not", [a]), ("not", [b]), ("and", ["g0", "g1"]), ("and", ["g0", b]), ("and", [a, "g1"]),
                ("and", [a, b])]
        ins = [a, b]
    return _nl("decoder2to4", ins, spec, [(outs[j], f"g{j + 2}") for j in range(4)])


def _half_subtractor(rng: random.Random) -> Netlist:
    a, b = rng.choice([("A", "B"), ("X", "Y"), ("a", "b")])
    d, bo = rng.choice([("D", "Bo"), ("Diff", "Borrow"), ("D", "B_out") if b != "B" else ("D", "Bout")])
    spec = [("xor", [a, b]), ("not", [a]), ("and", ["g1", b])]
    return _nl("half_subtractor", [a, b], spec, [(d, "g0"), (bo, "g2")])


def _comparator(rng: random.Random) -> Netlist:
    a, b = rng.choice([("A", "B"), ("a", "b"), ("X", "Y")])
    g, e, lt = rng.choice([("G", "E", "L"), (f"{a}>{b}", f"{a}={b}", f"{a}<{b}"), ("GT", "EQ", "LT")])
    if rng.random() < 0.5:
        spec = [("not", [a]), ("not", [b]), ("and", [a, "g1"]), ("and", ["g0", b]), ("xnor", [a, b])]
    else:
        spec = [("not", [a]), ("not", [b]), ("and", [a, "g1"]), ("and", ["g0", b]), ("nor", ["g2", "g3"])]
    return _nl("comparator1", [a, b], spec, [(g, "g2"), (e, "g4"), (lt, "g3")])


def _xor_from_nand(rng: random.Random) -> Netlist:
    a, b = rng.choice([("A", "B"), ("X", "Y"), ("a", "b")])
    y = rng.choice([s for s in ("Y", "F", "Q") if s not in (a, b)])
    spec = [("nand", [a, b]), ("nand", [a, "g0"]), ("nand", [b, "g0"]), ("nand", ["g1", "g2"])]
    return _nl("xor_from_nand", [a, b], spec, [(y, "g3")])


def _xnor_from_nor(rng: random.Random) -> Netlist:
    a, b = rng.choice([("A", "B"), ("X", "Y"), ("a", "b")])
    y = rng.choice([s for s in ("Y", "F", "Q") if s not in (a, b)])
    spec = [("nor", [a, b]), ("nor", [a, "g0"]), ("nor", [b, "g0"]), ("nor", ["g1", "g2"])]
    return _nl("xnor_from_nor", [a, b], spec, [(y, "g3")])


def _equality2(rng: random.Random) -> Netlist:
    ins = rng.choice([["A1", "A0", "B1", "B0"], ["A0", "A1", "B0", "B1"], ["a1", "a0", "b1", "b0"]])
    e = rng.choice(["E", "EQ", "Y", "A=B"])
    a1, a0 = sorted(ins[:2], reverse=True)
    b1, b0 = sorted(ins[2:], reverse=True)
    spec = [("xnor", [a0, b0]), ("xnor", [a1, b1]), ("and", ["g0", "g1"])]
    return _nl("equality2", ins, spec, [(e, "g2")])


def _adder2(rng: random.Random) -> Netlist:
    ins = rng.choice([["A0", "B0", "A1", "B1"], ["A0", "A1", "B0", "B1"], ["a0", "b0", "a1", "b1"]])
    a0, a1 = sorted([s for s in ins if s[0] in "Aa"])
    b0, b1 = sorted([s for s in ins if s[0] in "Bb"])
    s0, s1, c = rng.choice([("S0", "S1", "C"), ("S0", "S1", "Cout"), ("s0", "s1", "c2")])
    spec = [("xor", [a0, b0]), ("and", [a0, b0]),                    # half adder, bit 0
            ("xor", [a1, b1]), ("xor", ["g2", "g1"]), ("and", [a1, b1]), ("and", ["g2", "g1"]),
            ("or", ["g5", "g4"])]
    return _nl("adder2", ins, spec, [(s0, "g0"), (s1, "g3"), (c, "g6")])


CLASSICS: dict[str, Callable[[random.Random], Netlist]] = {
    "half_adder": _half_adder, "full_adder": _full_adder, "mux2": _mux2, "demux2": _demux2,
    "majority": _majority, "parity": _parity, "decoder2to4": _decoder2to4,
    "half_subtractor": _half_subtractor, "comparator1": _comparator, "xor_from_nand": _xor_from_nand,
    "xnor_from_nor": _xnor_from_nor, "equality2": _equality2, "adder2": _adder2,
}


def classic_netlist(rng: random.Random) -> Netlist:
    nl = CLASSICS[rng.choice(sorted(CLASSICS))](rng)
    nl.check()
    return nl
