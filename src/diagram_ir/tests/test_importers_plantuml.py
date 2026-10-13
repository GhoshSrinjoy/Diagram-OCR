"""Tests for the PlantUML importers (class, sequence, activity, state, use case, dispatcher).
Run:  python -m pytest src/diagram_ir/tests -q
"""

from __future__ import annotations

import pytest

from diagram_ir import Diagram, canonicalize, dumps, validate
from diagram_ir.importers.plantuml import PlantUMLError, UnsupportedPlantUML, plantuml_to_diagram
from diagram_ir.importers.plantuml_uml import detect_kind, preprocess


def parse(body: str) -> Diagram:
    d = plantuml_to_diagram(f"@startuml\n{body}\n@enduml\n")
    assert validate(d, strict=True) == []
    return d


def by_text(d: Diagram) -> dict[str, str]:
    """node text -> node id (and group text -> group id)."""
    out = {n.text: n.id for n in d.nodes}
    out.update({g.text: g.id for g in d.groups})
    return out


def edges(d: Diagram) -> list[tuple[str, str, str]]:
    """(source text, target text, kind) for every edge, in order."""
    names = {n.id: n.text for n in d.nodes} | {g.id: g.text for g in d.groups}
    return [(names[e.source], names[e.target], e.kind) for e in d.edges]


def group_of(d: Diagram, kind: str) -> list:
    return [g for g in d.groups if g.kind == kind]


# --------------------------------------------------------------------------- #
# Class diagrams
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("line, expected", [
    ("Animal <|-- Dog", ("Dog", "Animal", "inherit")),
    ("Dog --|> Animal", ("Dog", "Animal", "inherit")),
    ("Dog -up-|> Animal", ("Dog", "Animal", "inherit")),
    ("Shape <|.. Circle", ("Circle", "Shape", "realize")),
    ("Circle ..|> Shape", ("Circle", "Shape", "realize")),
    ("Car *-- Wheel", ("Car", "Wheel", "compose")),
    ("Wheel --* Car", ("Car", "Wheel", "compose")),
    ("Team o-- Player", ("Team", "Player", "aggregate")),
    ("Player --o Team", ("Team", "Player", "aggregate")),
    ("A --> B", ("A", "B", "assoc_dir")),
    ("A <-- B", ("B", "A", "assoc_dir")),
    ("A -[#red]-> B", ("A", "B", "assoc_dir")),
    ("A -l-> B", ("A", "B", "assoc_dir")),
    ("A ..> B", ("A", "B", "depend")),
    ("A <.. B", ("B", "A", "depend")),
    ("A -- B", ("A", "B", "assoc")),
    ("A <--> B", ("A", "B", "assoc")),
])
def test_class_arrow_direction_conventions(line, expected):
    d = parse(f"class A\nclass B\n{line}")
    assert edges(d) == [expected]


def test_class_cardinalities_follow_the_edge_direction():
    d = parse('Company "1" *-- "1..*" Department : has >')
    e = d.edges[0]
    ids = by_text(d)
    assert (e.source, e.target, e.kind) == (ids["Company"], ids["Department"], "compose")
    assert (e.source_label, e.target_label, e.label) == ("1", "1..*", "has")
    d = parse('Department "1..*" --* "1" Company : < owns')
    e = d.edges[0]
    assert (e.source, e.target) == (by_text(d)["Company"], by_text(d)["Department"])
    assert (e.source_label, e.target_label, e.label) == ("1", "1..*", "owns")


def test_class_declarations_members_and_kinds():
    d = parse("""
abstract class "Base Shape" as Base<T> <<Entity>> {
  - id : int
  .. getters ..
  + {abstract} area() : double
}
interface Drawable
enum Color {
  RED
  GREEN
  +rgb() : int
}
entity Order
Base : +name : String
""")
    n = {x.text: x for x in d.nodes}
    base = n["Base Shape<T>"]
    assert base.kind == "abstract" and base.stereotype == "Entity"
    assert [(m.kind, m.text) for m in base.members] == [
        ("attr", "- id : int"), ("op", "+ area() : double"), ("attr", "+name : String")]
    assert n["Drawable"].kind == "interface"
    assert [(m.kind, m.text) for m in n["Color"].members] == [
        ("lit", "RED"), ("lit", "GREEN"), ("op", "+rgb() : int")]
    assert n["Order"].kind == "class" and n["Order"].stereotype == "entity"


def test_class_packages_and_dotted_names_nest():
    d = parse("""
package app.model {
  class User
}
class app.service.UserService
namespace util {
  class Strings
}
app.service.UserService --> User
""")
    groups = {g.text: g for g in d.groups}
    assert set(groups) == {"app", "model", "service", "util"}
    assert groups["model"].parent == groups["app"].id
    assert groups["service"].parent == groups["app"].id
    assert groups["util"].kind == "namespace"
    ids = by_text(d)
    assert ids["User"] in groups["model"].members
    assert ids["UserService"] in groups["service"].members
    assert edges(d) == [("UserService", "User", "assoc_dir")]


def test_class_notes_hidden_links_and_extends():
    d = parse("""
class A extends B implements C, D
note left of A : important
note "floating" as N1
N1 .. B
A -[hidden]- D
""")
    assert sorted(edges(d)) == sorted([
        ("A", "B", "inherit"), ("A", "C", "realize"), ("A", "D", "realize"),
        ("important", "A", "anchor"), ("floating", "B", "anchor")])
    assert {n.text: n.kind for n in d.nodes}["important"] == "note"


def test_class_names_resolve_per_namespace_like_plantuml():
    d = parse("""
class BaseClass
namespace net.dummy {
  .BaseClass <|-- Person
  Meeting o-- Person
}
namespace net.foo {
  net.dummy.Person <|- Person
  net.dummy.Meeting o-- Person
}
BaseClass <|-- net.unused.Person
""")
    groups = {g.id: g.text for g in d.groups}
    home = {n: groups[g.id] for g in d.groups for n in g.members}
    named = [(n.text, home.get(n.id, "")) for n in d.nodes]
    assert sorted(named) == sorted([("BaseClass", ""), ("Person", "dummy"), ("Meeting", "dummy"),
                                    ("Person", "foo"), ("Person", "unused")])
    ids = {(n.text, home.get(n.id, "")): n.id for n in d.nodes}
    assert any(e.source == ids["Person", "foo"] and e.target == ids["Person", "dummy"] for e in d.edges)


def test_class_custom_namespace_separator():
    d = parse("set namespaceSeparator ::\nGnome::N::Top <|-- Gnome::GObject::Object\nGnome::N::Top : +x")
    assert edges(d) == [("Object", "Top", "inherit")]
    assert [g.text for g in d.groups] == ["Gnome", "N", "GObject"]
    assert [m.text for n in d.nodes if n.text == "Top" for m in n.members] == ["+x"]


def test_class_member_references_and_dotted_member_lines():
    d = parse("class Logger {\n+log()\n}\nenum Logger::Level\nUser --> Logger::log\nFoo.java : +x")
    # `Foo.java` is class `java` in package `Foo` (PlantUML's default separator), not a link Foo . java
    assert sorted(n.text for n in d.nodes) == ["Logger", "Logger::Level", "User", "java"]
    assert edges(d) == [("User", "Logger", "assoc_dir")]
    assert [m.text for n in d.nodes if n.text == "java" for m in n.members] == ["+x"]
    assert [g.text for g in d.groups] == ["Foo"]


def test_class_hide_members_and_remove():
    d = parse("class A {\n+x : int\n+f()\n}\nclass B\nhide methods\nremove B")
    assert [n.text for n in d.nodes] == ["A"]
    assert [m.text for m in d.nodes[0].members] == ["+x : int"]


# --------------------------------------------------------------------------- #
# Sequence diagrams
# --------------------------------------------------------------------------- #
def test_sequence_messages_in_order_with_kinds():
    d = parse("""
actor User
participant "Web Server" as W
database DB
User -> W : request
W ->> DB : async query
DB --> W : rows
W <- DB : callback
W -> W : self call
W ->x DB : kill
W -[#red]> User : colored
""")
    assert [(n.kind, n.text) for n in d.nodes] == [
        ("actor", "User"), ("participant", "Web Server"), ("database", "DB")]
    assert [(s, t, k) for s, t, k in edges(d)] == [
        ("User", "Web Server", "sync"), ("Web Server", "DB", "async"), ("DB", "Web Server", "reply"),
        ("DB", "Web Server", "sync"), ("Web Server", "Web Server", "sync"),
        ("Web Server", "DB", "destroy"), ("Web Server", "User", "sync")]
    assert [e.label for e in d.edges][:3] == ["request", "async query", "rows"]


def test_sequence_alt_else_becomes_operands_with_message_members():
    d = parse("""
A -> B : login
alt valid
  B -> A : ok
  loop 3 times
    A -> B : ping
  end
else invalid
  B -> A : error
end
A -> B : bye
""")
    alt = group_of(d, "alt")[0]
    ops = [g for g in d.groups if g.kind == "operand"]
    loop = group_of(d, "loop")[0]
    assert alt.members == [] and [o.parent for o in ops] == [alt.id, alt.id]
    assert [(o.text, o.members) for o in ops] == [("[valid]", ["e2"]), ("[invalid]", ["e4"])]
    assert loop.parent == ops[0].id and loop.text == "[3 times]" and loop.members == ["e3"]
    assert len(d.edges) == 5


def test_sequence_return_create_destroy_and_notes():
    d = parse("""
participant A
participant B
A -> B ++ : call
create C
B -> C : new
return result
note over A, B : spanning
A -> B : delete
destroy B
""")
    assert edges(d) == [
        ("A", "B", "sync"), ("B", "C", "create"), ("B", "A", "reply"),
        ("spanning", "A", "anchor"), ("spanning", "B", "anchor"), ("A", "B", "destroy")]
    assert d.edges[2].label == "result"
    assert [n.kind for n in d.nodes] == ["participant", "participant", "participant", "note"]


def test_sequence_box_and_boundary_messages():
    d = parse('box "Backend" #lightblue\nparticipant API\nparticipant DB\nend box\n'
              "[-> API : start\nAPI -> DB")
    box = group_of(d, "box")[0]
    assert box.text == "Backend" and box.members == [by_text(d)["API"], by_text(d)["DB"]]
    assert edges(d) == [("API", "DB", "sync")]  # the found message has no gate node


def test_sequence_return_without_activation_and_shorthands():
    d = parse("actor actor\nactor ++\n@api -> db : query\nreturn rows\ndb -> db : vacuum")
    assert [n.text for n in d.nodes] == ["actor", "@api", "db"]  # `actor ++` activates, declares nothing
    assert edges(d) == [("@api", "db", "sync"), ("db", "@api", "reply"), ("db", "db", "sync")]
    with pytest.raises(PlantUMLError):  # no arrow head: not a sequence message, and not a class link
        plantuml_to_diagram("@startuml\nparticipant A\nA -- B\n@enduml")


def test_sequence_canonical_keeps_time_order():
    d = canonicalize(parse("B -> A : 1\nA -> B : 2\nB -> A : 3"))
    assert [e.label for e in d.edges] == ["1", "2", "3"]


# --------------------------------------------------------------------------- #
# Activity diagrams
# --------------------------------------------------------------------------- #
def test_activity_if_else_has_decision_and_merge():
    d = parse("""
start
:Read input;
if (valid?) then (yes)
  :Process;
else (no)
  :Report error;
endif
:Done;
stop
""")
    kinds = {n.text or n.kind: n.kind for n in d.nodes}
    assert kinds["valid?"] == "decision" and "merge" in kinds and kinds["initial"] == "initial"
    assert (("valid?", "Process", "flow") in edges(d)) and ("valid?", "Report error", "flow") in edges(d)
    labels = {(s, t): e.label for (s, t, _), e in zip(edges(d), d.edges, strict=True)}
    assert labels[("valid?", "Process")] == "yes" and labels[("valid?", "Report error")] == "no"
    assert ("Process", "", "flow") in edges(d) and ("Report error", "", "flow") in edges(d)
    assert ("", "Done", "flow") in edges(d)


def test_activity_if_without_else_and_with_stop():
    d = parse("start\nif (c) then (yes)\n:A;\nendif\n:B;")
    assert ("c", "", "flow") in edges(d) and ("A", "", "flow") in edges(d)  # c --> merge <-- A
    d = parse("start\nif (c) then (yes)\n:A;\nstop\nelse (no)\nendif\n:B;")
    assert not [n for n in d.nodes if n.kind == "merge"]  # only one branch continues
    assert ("c", "B", "flow") in edges(d) and d.edges[edges(d).index(("c", "B", "flow"))].label == "no"


def test_activity_elseif_chain_and_switch():
    d = parse("start\nif (a) then (1)\n:X;\nelseif (b) then (2)\n:Y;\nelse (3)\n:Z;\nendif\nstop")
    assert ("a", "b", "flow") in edges(d)
    assert sum(1 for s, t, _ in edges(d) if t == "" and s in ("X", "Y", "Z")) == 3
    d = parse("start\nswitch (x?)\ncase (A)\n:a;\ncase (B)\n:b;\nendswitch\nstop")
    assert [e.label for e in d.edges if e.source == by_text(d)["x?"]] == ["A", "B"]


def test_activity_swimlanes_loops_and_fork():
    d = parse("""
|Customer|
start
:Order;
|#AntiqueWhite|Shop|
while (items left?) is (yes)
  :Pack item;
endwhile (no)
fork
  :Ship;
fork again
  :Invoice;
end fork
repeat
  :Retry;
repeat while (failed?) is (yes) not (no)
stop
""")
    lanes = {g.text: g for g in group_of(d, "lane")}
    ids = by_text(d)
    assert ids["Order"] in lanes["Customer"].members and ids["Pack item"] in lanes["Shop"].members
    assert ("Pack item", "items left?", "flow") in edges(d)  # loop back edge
    assert {n.kind for n in d.nodes} >= {"fork", "join", "merge", "decision", "final"}
    assert ("failed?", "", "flow") in edges(d)  # repeat loops back to its merge


def test_activity_arrow_labels_notes_and_kill():
    d = parse("start\n:A;\n-> go on;\n:B;\nnote right: why\nkill\n:C;")
    assert ("A", "B", "flow") in edges(d) and d.edges[1].label == "go on"
    assert ("why", "B", "anchor") in edges(d)
    assert not any(t == "C" for _, t, _ in edges(d))  # kill ends the flow


def test_activity_repeat_label_and_blocks_closed_at_the_end():
    d = parse("start\nrepeat :Enter data;\n:Check;\nrepeat while (bad?) is (yes)\nif (more?) then (yes)\n:A;")
    assert not any(n.kind == "merge" and n.id == d.edges[0].target for n in d.nodes)
    assert ("bad?", "Enter data", "flow") in edges(d)  # the label box is the loop entry
    assert ("more?", "", "flow") in edges(d) and ("A", "", "flow") in edges(d)  # endif implied


def test_activity_legacy_syntax():
    d = parse("""
(*) --> "First"
--> [ok] "Second" as S
if "again?" then
  -->[yes] "First"
else
  -->[no] (*)
endif
""")
    ids = by_text(d)
    assert {n.kind for n in d.nodes} == {"initial", "action", "decision", "final"}
    assert ("First", "Second", "flow") in edges(d) and ("Second", "again?", "flow") in edges(d)
    assert ("again?", "First", "flow") in edges(d) and ids["again?"]


# --------------------------------------------------------------------------- #
# State diagrams
# --------------------------------------------------------------------------- #
def test_state_initial_final_are_per_scope_and_composites_are_groups():
    d = parse("""
[*] --> Idle
Idle --> Active : start
state Active {
  [*] --> Working
  Working --> [*]
}
Active --> [*] : stop
state "Long name" as L
state Pick <<choice>>
Idle --> L
Idle --> Pick
""")
    comp = group_of(d, "composite")[0]
    assert comp.text == "Active"
    initials = [n for n in d.nodes if n.kind == "initial"]
    finals = [n for n in d.nodes if n.kind == "final"]
    assert len(initials) == 2 and len(finals) == 2
    inner_initial = next(n for n in initials if n.id in comp.members)
    assert inner_initial.id in comp.members and by_text(d)["Working"] in comp.members
    assert ("Idle", "Active", "transition") in edges(d) and ("Active", "", "transition") in edges(d)
    assert {n.text: n.kind for n in d.nodes}["Long name"] == "state"
    assert any(n.kind == "choice" for n in d.nodes)


def test_state_dotted_descriptions_and_composite_descriptions():
    d = parse("[*] --> App.java\nApp.java : entry point\nstate Busy {\n[*] --> Work\n}\nBusy : do / work\n"
              "App.java --> Busy")
    assert {n.text: [m.text for m in n.members] for n in d.nodes if n.kind == "state"} == {
        "App.java": ["entry point"], "Work": []}
    assert group_of(d, "composite")[0].text == "Busy\ndo / work"


def test_state_concurrent_regions():
    d = parse("state S {\n[*] --> A\n--\n[*] --> B\n}")
    regions = group_of(d, "region")
    comp = group_of(d, "composite")[0]
    assert len(regions) == 2 and all(r.parent == comp.id for r in regions)
    ids = by_text(d)
    assert ids["A"] in regions[0].members and ids["B"] in regions[1].members
    assert sum(n.kind == "initial" for n in d.nodes) == 2


# --------------------------------------------------------------------------- #
# Use case diagrams
# --------------------------------------------------------------------------- #
def test_usecase_actors_usecases_and_boundary():
    d = parse("""
left to right direction
actor :Customer: as C
rectangle Shop {
  usecase "Check out" as UC1
  (Browse)
}
C --> UC1
C -- (Browse)
User <|-- C
""")
    kinds = {n.text: n.kind for n in d.nodes}
    assert d.type == "graph" and d.direction == "LR"
    assert kinds == {"Customer": "actor", "Check out": "ellipse", "Browse": "ellipse", "User": "actor"}
    assert group_of(d, "cluster")[0].members == [by_text(d)["Check out"], by_text(d)["Browse"]]
    assert edges(d) == [("Customer", "Check out", "arrow"), ("Customer", "Browse", "line"),
                        ("Customer", "User", "arrow")]


# --------------------------------------------------------------------------- #
# Preprocessing and dispatch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code, kind", [
    ("class A\nA <|-- B", "class"),
    ("Alice -> Bob : hi", "sequence"),
    ("start\n:a;\nstop", "activity"),
    ('(*) --> "a"', "activity_legacy"),
    ("[*] --> A", "state"),
    ("actor U\nU --> (Use)", "usecase"),
    ("[Web] --> [DB]", "component"),
])
def test_detect_kind(code, kind):
    assert detect_kind(f"@startuml\n{code}\n@enduml") == kind


@pytest.mark.parametrize("code", [
    "@startgantt\n[Task] lasts 5 days\n@endgantt",
    "@startmindmap\n* root\n@endmindmap",
    "@startuml\nrobust \"Web\" as W\n@W\nW is Idle\n@enduml",
    "@startuml\nsalt\n{\n[OK]\n}\n@enduml",
])
def test_non_uml_diagrams_are_unsupported(code):
    with pytest.raises(UnsupportedPlantUML):
        plantuml_to_diagram(code)


def test_unparseable_input_raises():
    with pytest.raises(PlantUMLError):
        plantuml_to_diagram("@startuml\nclass A {\n}\nA ~~~ ??? !!\n@enduml")


def test_preprocessing_comments_macros_first_block_and_title():
    code = """```plantuml
@startuml
' a comment
/' block
comment '/
!define ENT(n) class n <<entity>>
!include common.iuml
title My Model
skinparam class {
  BackgroundColor red
}
ENT(User)
note right of User
  multi
  line
end note
@enduml
@startuml
class Ignored
@enduml
```"""
    src = preprocess(code)
    assert src.title == "My Model" and src.macro_lines == 1
    d = plantuml_to_diagram(code)
    assert [(n.text, n.kind, n.stereotype) for n in d.nodes] == [
        ("User", "class", "entity"), ("multi\nline", "note", None)]
    assert validate(d, strict=True) == []


def test_output_is_canonicalizable():
    d = parse("class A\nclass B\nA <|-- B\nA *-- C")
    text = dumps(canonicalize(d))
    assert "edge" in text and validate(canonicalize(d), strict=True) == []
