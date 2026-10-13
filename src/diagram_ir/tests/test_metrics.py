"""Tests for diagram_ir.metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from diagram_ir.metrics import aggregate, score

FLOW = (Path(__file__).resolve().parents[1] / "examples" / "flowchart.dgm").read_text(encoding="utf-8")


def test_perfect_prediction():
    r = score(FLOW, FLOW)
    assert r["exact"] == 1 and r["node_f1"] == 1.0 and r["edge_f1"] == 1.0 and r["valid"] == 1


def test_ids_and_order_do_not_matter():
    gold = 'diagram flowchart\nnode n1 terminal "Start"\nnode n2 process "Work"\nedge n1 n2\n'
    pred = 'diagram flowchart\nnode b process "Work"\nnode a terminal "Start"\nedge a b\n'
    r = score(pred, gold)
    assert r["node_f1"] == 1.0 and r["edge_f1"] == 1.0


def test_reversed_edge_is_wrong_but_nodes_right():
    gold = 'diagram flowchart\nnode n1 process "A"\nnode n2 process "B"\nedge n1 n2\n'
    pred = 'diagram flowchart\nnode n1 process "A"\nnode n2 process "B"\nedge n2 n1\n'
    r = score(pred, gold)
    assert r["node_f1"] == 1.0 and r["edge_f1"] == 0.0


def test_undirected_edge_direction_ignored():
    gold = 'diagram class\nnode n1 class "A"\nnode n2 class "B"\nedge n1 n2\n'
    pred = 'diagram class\nnode n1 class "A"\nnode n2 class "B"\nedge n2 n1\n'
    assert score(pred, gold)["edge_f1"] == 1.0


def test_typo_still_matches_and_label_metric_separate():
    gold = 'diagram flowchart\nnode n1 decision "Quality Check"\nnode n2 process "Ship"\nedge n1 n2 "Pass"\n'
    pred = 'diagram flowchart\nnode n1 decision "Qualty Check"\nnode n2 process "Ship"\nedge n1 n2 "Fail"\n'
    r = score(pred, gold)
    assert r["node_f1"] == 1.0 and r["text_exact"] == 1 and r["edge_f1"] == 1.0 and r["edge_label_f1"] == 0.0


def test_garbage_and_truncated_output():
    gold = 'diagram flowchart\nnode n1 process "A"\nnode n2 process "B"\nedge n1 n2\n'
    r = score("I cannot read this image.", gold)
    assert r["valid"] == 0 and r["node_f1"] == 0.0
    r = score('diagram flowchart\nnode n1 process "A"\nnode n2 process "B"\nedge n1 n', gold)
    assert r["node_f1"] == 1.0 and r["edge_r"] == 0.0


def test_text_free_circuit_matching_uses_kinds():
    gold = "diagram circuit\nnode n1 and\nnode n2 not\nedge n1 n2\n"
    pred = "diagram circuit\nnode n1 and\nnode n2 not\nedge n1 n2\n"
    assert score(pred, gold)["edge_f1"] == 1.0


def test_identical_gates_are_paired_by_wiring():
    # Three text-free OR gates: the gold fans out from one gate, the prediction fans in. At best one
    # edge agrees; pairing the gates by degree and position alone found none.
    gold = "diagram circuit\nnode n1 or\nnode n2 or\nnode n3 or\nedge n1 n2\nedge n1 n3\n"
    pred = "diagram circuit\nnode n1 or\nnode n3 or\nnode n2 or\nedge n1 n2\nedge n3 n2\n"
    r = score(pred, gold)
    assert r["node_f1"] == 1.0 and r["edge_hits"] == 1 and r["edge_f1"] == 0.5


def test_edges_to_groups_are_scored():
    # A message flow from a collapsed pool (a group with no members) into the process.
    gold = ('diagram activity\nnode n1 initial\nnode n2 action "Check claim"\nedge n1 n2\nedge g2 n2\n'
            'group g1 lane "Insurer" : n1 n2\ngroup g2 lane "Customer"\n')
    pred = ('diagram activity\nnode n1 initial\nnode n2 action "Check claim"\nedge n1 n2\nedge g1 n2\n'
            'group g1 lane "Customer"\ngroup g2 lane "Insurer" : n1 n2\n')
    assert score(gold, gold)["edge_f1"] == 1.0
    assert score(pred, gold)["edge_f1"] == 1.0
    wrong = pred.replace("edge g1 n2", "edge g2 n2")  # the flow leaves the Insurer lane instead
    assert score(wrong, gold)["edge_hits"] == 1


def test_aggregate_micro_average():
    gold = 'diagram flowchart\nnode n1 process "A"\nnode n2 process "B"\nedge n1 n2\n'
    rs = [score(gold, gold), score('diagram flowchart\nnode n1 process "A"\n', gold)]
    agg = aggregate(rs)
    assert agg["n"] == 2 and agg["node_r"] == pytest.approx(3 / 4) and agg["edge_r"] == pytest.approx(1 / 2)
