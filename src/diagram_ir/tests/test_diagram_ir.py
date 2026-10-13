"""Tests for the Diagram IR: parsing, round-trips through every format, canonical form,
validation and the CLI.  Run:  python -m pytest src/diagram_ir/tests -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagram_ir import (
    DgmSyntaxError,
    Diagram,
    Edge,
    Node,
    canonicalize,
    detect_format,
    dump,
    dumps,
    errors,
    from_dict,
    load,
    loads,
    parse,
    to_dict,
    validate,
)
from diagram_ir.__main__ import main as cli_main

PKG = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((PKG / "examples").glob("*.dgm"))


def _body(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("#")) + "\n"


@pytest.fixture(params=EXAMPLES, ids=lambda p: p.stem)
def example(request) -> tuple[str, Diagram]:
    text = request.param.read_text(encoding="utf-8")
    return text, loads(text)


# --------------------------------------------------------------------------- #
# Examples: valid, canonical, lossless through every format
# --------------------------------------------------------------------------- #
def test_examples_cover_main_types():
    types = {loads(p.read_text(encoding="utf-8")).type for p in EXAMPLES}
    assert {"flowchart", "class", "sequence", "activity", "state", "circuit"} <= types


def test_example_is_strictly_valid(example):
    _, d = example
    assert validate(d, strict=True) == []


def test_example_is_already_canonical(example):
    text, d = example
    assert dumps(canonicalize(d)) == _body(text)


@pytest.mark.parametrize("fmt", ["json", "yaml", "xml"])
def test_example_round_trips(example, fmt):
    text, d = example
    assert dumps(load(dump(d, fmt), fmt)) == _body(text)


def test_example_json_matches_schema(example):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((PKG / "schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(to_dict(example[1]), schema)


def test_canonicalize_is_idempotent_with_and_without_layout(example):
    _, d = example
    for variant in (d, d.strip_layout()):
        once = canonicalize(variant)
        assert dumps(canonicalize(once)) == dumps(once)


# --------------------------------------------------------------------------- #
# Canonical form
# --------------------------------------------------------------------------- #
def test_canonical_order_follows_arrows_not_source_order():
    # Same graph as the flowchart example, written in scrambled order with Mermaid ids.
    text = """diagram flowchart
node Z terminal "End"
node Q process "Review"
node A terminal "Start"
node M decision "Check"
node K process "Work"
node P process "Done"
edge Q K
edge M P "Pass"
edge A K
edge K M
edge M Q "Fail"
edge P Z
"""
    out = dumps(canonicalize(loads(text)))
    assert out == """diagram flowchart
node n1 terminal "Start"
node n2 process "Work"
node n3 decision "Check"
node n4 process "Done"
node n5 process "Review"
node n6 terminal "End"
edge n1 n2
edge n2 n3
edge n3 n4 "Pass"
edge n3 n5 "Fail"
edge n4 n6
edge n5 n2
"""


def test_canonical_branch_order_uses_layout_when_present():
    # Fail branch drawn on the LEFT -> it must come first, regardless of edge order.
    text = """diagram flowchart
node d decision "OK?" @450,100,550,200
node y process "Yes path" @700,300,900,350
node n process "No path" @100,300,300,350
edge d y "yes"
edge d n "no"
"""
    out = dumps(canonicalize(loads(text)))
    assert 'node n2 process "No path"' in out and 'node n3 process "Yes path"' in out


def test_reading_order_rows_with_unequal_heights():
    text = """diagram class
node b class "B" @400,40,600,300
node c class "C" @700,80,900,160
node a class "A" @50,60,250,200
node d class "D" @50,500,250,600
"""
    out = dumps(canonicalize(loads(text)))
    names = [ln.split('"')[1] for ln in out.splitlines() if ln.startswith("node")]
    assert names == ["A", "B", "C", "D"]


def test_sequence_keeps_message_order_and_remaps_fragments():
    text = """diagram sequence
node b participant "B" @500,10,600,60
node a participant "A" @100,10,200,60
edge a b "first"
edge b a reply "second"
group f loop "[retry]" : e2
"""
    out = dumps(canonicalize(loads(text)))
    assert out.splitlines()[3:] == [
        'edge n1 n2 "first"',
        'edge n2 n1 reply "second"',
        'group g1 loop "[retry]" : e2',
    ]


def test_canonical_text_whitespace():
    d = loads('diagram flowchart\nnode a process "  M = 1 \\n\\n  F   =  1  "\n')
    assert canonicalize(d).nodes[0].text == "M = 1\n\nF = 1"


def test_edges_into_groups_count_as_flow():
    d = canonicalize(loads((PKG / "examples" / "state.dgm").read_text(encoding="utf-8")))
    texts = [n.text or n.kind for n in d.nodes]
    assert texts == ["initial", "Idle", "initial", "Validating", "Charging", "final"]


# --------------------------------------------------------------------------- #
# DGM syntax
# --------------------------------------------------------------------------- #
def test_escapes_and_unicode_round_trip():
    tricky = 'Say "hi" \\ to\nthe Größe → 世界 & <tags>'
    d = Diagram(type="flowchart", nodes=[Node(id="n1", kind="process", text=tricky)])
    for fmt in ("dgm", "json", "yaml", "xml"):
        assert load(dump(d, fmt), fmt).nodes[0].text == tricky, fmt


def test_yaml_keeps_ambiguous_scalars_as_strings():
    d = Diagram(type="flowchart", nodes=[Node(id=t, kind="process", text=t) for t in ("yes", "no")]
                + [Node(id="n3", kind="process", text="1"), Node(id="n4", kind="process", text="null")])
    back = load(dump(d, "yaml"), "yaml")
    assert [n.text for n in back.nodes] == ["yes", "no", "1", "null"]


def test_default_edge_kind_omitted_and_restored():
    d = Diagram(type="class", nodes=[Node("a", "class", "A"), Node("b", "class", "B")],
                edges=[Edge("a", "b", "assoc"), Edge("a", "b", "inherit")])
    text = dumps(d)
    assert "edge a b\n" in text and "edge a b inherit\n" in text
    assert [e.kind for e in loads(text).edges] == ["assoc", "inherit"]


def test_attr_values_quoted_only_when_needed():
    d = loads('diagram class\nnode a class "A"\nnode b class "B"\n'
              'edge a b sl="0..*" tl="one or more" x_note=ok\n')
    line = dumps(d).splitlines()[-1]
    assert line == 'edge a b sl=0..* tl="one or more" x_note=ok'
    assert d.edges[0].attrs == {"x_note": "ok"}


def test_path_attribute():
    d = loads('diagram graph\nnode a node\nnode b node\nedge a b path=10,20,10,80,90,80\n')
    assert d.edges[0].path == [(10, 20), (10, 80), (90, 80)]
    assert "path=10,20,10,80,90,80" in dumps(d)
    assert to_dict(d)["edges"][0]["path"] == [[10, 20], [10, 80], [90, 80]]


def test_strict_parse_reports_line_number():
    with pytest.raises(DgmSyntaxError) as exc:
        loads('diagram flowchart\nnode n1 process "ok"\nnode n2 process "unterminated\n')
    assert exc.value.line_no == 3


def test_lenient_parse_salvages_truncated_output():
    raw = ('```\ndiagram flowchart\nnode n1 terminal "Start"\nnode n2 process "Work"\n'
           'edge n1 n2\nedge n2 n3 "Tr')
    d, errs = parse(raw)
    assert [n.id for n in d.nodes] == ["n1", "n2"] and len(d.edges) == 1
    assert len(errs) == 1 and errs[0].line_no == 6


def test_lenient_parse_without_header():
    d, errs = parse('node n1 process "A"\nnode n2 process "B"\nedge n1 n2\n')
    assert d.type == "graph" and len(d.nodes) == 2 and len(errs) == 1


def test_member_must_follow_node():
    with pytest.raises(DgmSyntaxError):
        loads('diagram class\nattr "-x: int"\n')


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _messages(d: Diagram, strict: bool = False) -> list[str]:
    return [str(i) for i in validate(d, strict=strict)]


def test_validation_catches_structural_errors():
    d = loads("""diagram flowchart
node n1 process "A" @10,10,5,20
node n1 process "dup"
node e7 process "reserved"
edge n1 ghost
group g1 lane : n1 e9 g2
group g2 lane parent=g3
group g3 lane parent=g2
""")
    msgs = "\n".join(_messages(d))
    for needle in ("x1<=x2", "duplicate id", "reserved for edges", "'ghost' does not exist",
                   "missing edge", "nest with parent=", "cycle"):
        assert needle in msgs, needle


def test_vocabulary_is_warning_unless_strict():
    d = loads('diagram flowchart\nnode n1 hexagon "A"\n')
    assert [i.level for i in validate(d)] == ["warning"]
    assert [i.level for i in validate(d, strict=True)] == ["error"]


def test_members_on_unusual_kind_warn():
    d = loads('diagram flowchart\nnode n1 process "A"\n  attr "x"\n')
    issues = validate(d)
    assert issues and not errors(issues)


def test_bbox_out_of_range():
    d = loads('diagram flowchart\nnode n1 process @0,0,1001,10\n')
    assert any("0..1000" in m for m in _messages(d))


# --------------------------------------------------------------------------- #
# JSON / YAML / XML specifics
# --------------------------------------------------------------------------- #
def test_from_dict_remaps_custom_edge_ids_and_defaults():
    d = from_dict({
        "type": "sequence",
        "nodes": [{"id": "a", "kind": "participant", "text": "A"},
                  {"id": "b", "kind": "participant", "text": "B"}],
        "edges": [{"id": "m_hello", "source": "a", "target": "b", "label": "hello"},
                  {"id": "m_bye", "source": "b", "target": "a", "kind": "reply"}],
        "groups": [{"id": "g", "kind": "opt", "members": ["m_bye"]}],
    })
    assert d.edges[0].kind == "sync" and d.groups[0].members == ["e2"]
    assert validate(d, strict=True) == []


def test_from_dict_error_has_field_path():
    with pytest.raises(ValueError, match=r"\$\.nodes\[0\]\.text: expected str"):
        from_dict({"type": "flowchart", "nodes": [{"id": "n1", "kind": "process", "text": True}]})


def test_json_marks_direction_of_edges():
    d = loads('diagram class\nnode a class\nnode b class\nedge a b\nedge a b inherit\n')
    assert [e["directed"] for e in to_dict(d)["edges"]] == [False, True]


def test_detect_format():
    assert detect_format("# c\ndiagram flowchart\n") == "dgm"
    assert detect_format('{"type": "graph"}') == "json"
    assert detect_format("<?xml version='1.0'?><diagram/>") == "xml"
    assert detect_format("type: graph\nnodes: []\n") == "yaml"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_convert_and_validate(tmp_path, capsys):
    src = PKG / "examples" / "flowchart.dgm"
    out_dir = tmp_path / "exports"
    assert cli_main(["convert", str(src), "--to", "json,yaml,xml", "-o", str(out_dir)]) == 0
    for ext in ("json", "yaml", "xml"):
        exported = out_dir / f"flowchart.{ext}"
        assert dumps(load(exported.read_text(encoding="utf-8"))) == _body(src.read_text(encoding="utf-8"))
    assert cli_main(["validate", str(out_dir / "flowchart.xml"), "--strict"]) == 0
    assert "0 errors" in capsys.readouterr().out


def test_undirected_edges_stored_from_earlier_endpoint():
    a = loads('diagram class\nnode x class "B" @500,10,600,60\nnode y class "A" @10,10,100,60\n'
              'edge x y sl=1 tl=0..*\n')
    out = dumps(canonicalize(a))
    assert "edge n1 n2 sl=0..* tl=1" in out  # A (n1) -- B (n2), end labels swapped with it


def test_idempotent_with_container_entry_and_undirected_lines():
    text = """diagram state
node s state "Idle"
node i initial
node v state "Validating"
node c state "Charging"
node t note "slow"
edge s g "start"
edge g s "cancel"
edge i v
edge v c
edge t c anchor
group g composite "Active" : c v i
"""
    once = canonicalize(loads(text))
    assert dumps(canonicalize(once)) == dumps(once)
    assert dumps(canonicalize(once.strip_layout())) == dumps(once.strip_layout())
