"""Tests for the Mermaid importers (flowchart fixes, sequence, class, state, ER, mindmap, dispatch).
Run:  python -m pytest src/diagram_ir/tests -q
"""

from __future__ import annotations

import pytest

from diagram_ir import Diagram, canonicalize, dumps, validate
from diagram_ir.importers.mermaid import (
    MermaidError,
    UnsupportedMermaid,
    class_member_text,
    mermaid_flowchart_to_diagram,
    mermaid_to_diagram,
)


def _parse(code: str) -> Diagram:
    d = mermaid_to_diagram(code)
    assert validate(d, strict=True) == []
    assert validate(canonicalize(d), strict=True) == []
    return d


def _node(d: Diagram, text: str):
    return next(n for n in d.nodes if n.text == text)


def _ids(d: Diagram, *texts: str) -> list[str]:
    return [_node(d, t).id for t in texts]


def _edge_tuples(d: Diagram) -> list[tuple]:
    by_id = {n.id: n.text for n in d.nodes} | {g.id: g.text for g in d.groups}
    return [(by_id[e.source], by_id[e.target], e.kind, e.label) for e in d.edges]


# --------------------------------------------------------------------------- #
# Flowchart regressions
# --------------------------------------------------------------------------- #
def test_flowchart_node_text_wrapped_over_lines():
    d = mermaid_flowchart_to_diagram(
        "graph TD\n    A[Start] --> B{Is there any\n  need for X?}\n"
        "    B -->|Yes| C[Analyze the\nprocurement data]\n"
        "    %% comment inside\n    C --> D[End]"
    )
    assert [n.text for n in d.nodes] == ["Start", "Is there any\nneed for X?",
                                         "Analyze the\nprocurement data", "End"]
    assert _node(d, "Is there any\nneed for X?").kind == "decision"
    assert len(d.edges) == 3


def test_flowchart_unbalanced_bracket_does_not_swallow_the_rest():
    # An unclosed bracket is an error on its own line, not a reason to eat every later line.
    with pytest.raises(MermaidError):
        mermaid_flowchart_to_diagram("graph TD\nA[oops --> B\nB --> C")


def test_flowchart_space_between_quote_and_closer():
    d = mermaid_flowchart_to_diagram("""flowchart TD
    E1[/"Output 'x'" /] --> B["text" ]""")
    assert [(n.kind, n.text) for n in d.nodes] == [("io", "Output 'x'"), ("process", "text")]


def test_flowchart_escaped_quotes_in_quoted_text():
    d = mermaid_flowchart_to_diagram(
        'flowchart TD\n    A(["Start"]) --> B[/"12\\" Powerbook and Tools"/]\n    B --> C["Done"]'
    )
    assert _node(d, '12" Powerbook and Tools').kind == "io"
    assert len(d.nodes) == 3 and len(d.edges) == 2


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code, dtype", [
    ("graph LR\nA-->B", "flowchart"),
    ("---\ntitle: T\n---\nflowchart TD\nA-->B", "flowchart"),
    ("sequenceDiagram\nA->>B: hi", "sequence"),
    ("classDiagram\nA <|-- B", "class"),
    ("stateDiagram-v2\n[*] --> A", "state"),
    ("stateDiagram\n[*] --> A", "state"),
    ("erDiagram\nA ||--o{ B : has", "er"),
    ("%%{init: {'theme': 'dark'}}%%\nmindmap\n  root((R))\n    A", "graph"),
])
def test_dispatch_routes_on_header(code, dtype):
    assert _parse(code).type == dtype


@pytest.mark.parametrize("code", ["pie\n\"a\": 1", "gantt\ntitle x", "gitGraph\ncommit",
                                  "journey\ntitle x", "C4Dynamic\ntitle x", "zenuml\nA.b()",
                                  "block-beta\nA", "nonsense"])
def test_dispatch_rejects_other_types(code):
    with pytest.raises(UnsupportedMermaid):
        mermaid_to_diagram(code)


def test_dispatch_empty():
    with pytest.raises(MermaidError):
        mermaid_to_diagram("  \n%% only a comment\n")


# --------------------------------------------------------------------------- #
# Sequence diagrams
# --------------------------------------------------------------------------- #
def test_sequence_participants_messages_and_kinds():
    d = _parse("""---
title: Login
---
sequenceDiagram
    autonumber
    actor U as User
    participant API as Backend API
    participant DB@{ "type": "database" }
    U->>+API: login(user)
    activate API
    API-->>-U: token
    API-)DB: audit
    DB--)API: ack
    API->API: self check
    API-xDB: drop
    API--xDB: dropped
    API->>DB: a; DB-->>API: b
""")
    assert d.title == "Login"
    assert [(n.kind, n.text) for n in d.nodes] == [
        ("actor", "User"), ("participant", "Backend API"), ("database", "DB")]
    u, api, db = (n.id for n in d.nodes)
    assert [(e.source, e.target, e.kind, e.label, e.style) for e in d.edges] == [
        (u, api, "sync", "login(user)", None),
        (api, u, "reply", "token", None),
        (api, db, "async", "audit", None),
        (db, api, "reply", "ack", None),
        (api, api, "sync", "self check", None),  # self message
        (api, db, "destroy", "drop", None),
        (api, db, "destroy", "dropped", "dashed"),
        (api, db, "sync", "a", None),
        (db, api, "reply", "b", None),
    ]


def test_sequence_first_use_order():
    d = _parse("sequenceDiagram\nB->>A: x\nparticipant C\nA->>C: y")
    assert [n.text for n in d.nodes] == ["B", "A", "C"]


def test_sequence_alt_with_branches_becomes_operands_holding_message_ids():
    d = _parse("""sequenceDiagram
    A->>B: start
    loop Every minute
        A->>B: ping
        alt is ok
            B-->>A: pong
        else is slow
            B-->>A: late
            B-->>A: sorry
        end
    end
    opt maybe
        A->>B: extra
    end
""")
    g = {gr.kind: gr for gr in d.groups if gr.kind != "operand"}
    ops = [gr for gr in d.groups if gr.kind == "operand"]
    assert g["loop"].text == "[Every minute]" and g["loop"].members == ["e2"] and g["loop"].parent is None
    assert g["alt"].text == "" and g["alt"].parent == g["loop"].id and g["alt"].members == []
    assert [(o.text, o.members, o.parent) for o in ops] == [
        ("[is ok]", ["e3"], g["alt"].id), ("[is slow]", ["e4", "e5"], g["alt"].id)]
    assert g["opt"].text == "[maybe]" and g["opt"].members == ["e6"]
    # positional edge ids survive canonicalization (time order is kept)
    c = canonicalize(d)
    assert [e.label for e in c.edges] == [e.label for e in d.edges]


def test_sequence_par_critical_break_rect():
    d = _parse("""sequenceDiagram
    par Alice to Bob
        A->>B: hi
    and Alice to John
        A->>J: hi
    end
    critical Connect
        A->>DB: connect
    option Timeout
        A->>A: log
    end
    break when failing
        A->>B: fail
    end
    rect rgb(191, 223, 255)
        A->>B: in rect
    end
    critical Alone
        A->>B: only
    end
""")
    kinds = [(g.kind, g.text, g.members) for g in d.groups]
    assert kinds == [
        ("par", "", []), ("operand", "[Alice to Bob]", ["e1"]), ("operand", "[Alice to John]", ["e2"]),
        ("critical", "", []), ("operand", "[Connect]", ["e3"]), ("operand", "[Timeout]", ["e4"]),
        ("break", "[when failing]", ["e5"]), ("group", "", ["e6"]),
        ("critical", "[Alone]", ["e7"]),  # a single branch needs no operand group
    ]


def test_sequence_notes_are_anchored_in_time_order():
    d = _parse("""sequenceDiagram
    participant A
    participant B
    A->>B: one
    Note right of A: thinking<br/>hard
    Note over A,B: shared
    alt ok
        note left of B: inside
        B-->>A: two
    end
""")
    notes = [n for n in d.nodes if n.kind == "note"]
    assert [n.text for n in notes] == ["thinking\nhard", "shared", "inside"]
    a, b = _ids(d, "A", "B")
    assert [(e.source, e.target, e.kind) for e in d.edges] == [
        (a, b, "sync"), (notes[0].id, a, "anchor"), (notes[1].id, a, "anchor"), (notes[1].id, b, "anchor"),
        (notes[2].id, b, "anchor"), (b, a, "reply")]
    alt = next(g for g in d.groups if g.kind == "alt")
    assert alt.members == [notes[2].id, "e6"]  # note node + message, not the anchor edge


def test_sequence_box_create_destroy():
    d = _parse("""sequenceDiagram
    box Aqua Frontend team
        participant W as Web
        actor U
    end
    box Backend
        participant S
    end
    W->>S: request
    create participant C as Cache
    S->>C: new
    C-->>S: ok
    destroy C
    S-xC: bye
""")
    boxes = [g for g in d.groups if g.kind == "box"]
    assert [(b.text, b.members) for b in boxes] == [
        ("Frontend team", _ids(d, "Web", "U")), ("Backend", _ids(d, "S"))]
    assert [e.kind for e in d.edges] == ["sync", "create", "reply", "destroy"]


@pytest.mark.parametrize("code", [
    "sequenceDiagram\nA->>B: x\nend",
    "sequenceDiagram\nloop forever\nA->>B: x",
    "sequenceDiagram\nelse nope\nA->>B: x",
    "sequenceDiagram\nthis is not a statement",
])
def test_sequence_malformed(code):
    with pytest.raises(MermaidError):
        mermaid_to_diagram(code)


# --------------------------------------------------------------------------- #
# Class diagrams
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, expected", [
    ("+String name", (False, "+String name")),
    ("-List~int~ ids", (False, "-List<int> ids")),
    ("+count$", (False, "+count")),
    ("+total() float", (True, "+total() : float")),
    ("+find(id: int) List~List~T~~", (True, "+find(id: int) : List<List<T>>")),
    ("#run()*", (True, "#run()")),
    ("~helper(a, b) int$", (True, "~helper(a, b) : int")),
    ("someMethod()", (True, "someMethod()")),
])
def test_class_member_display_text(raw, expected):
    assert class_member_text(raw) == expected


def test_class_relation_directions_and_cardinalities():
    d = _parse("""classDiagram
    Animal <|-- Duck
    Fish --|> Animal
    Car *-- "4" Wheel
    Wheel "1..*" --* "1" Car2
    Pond o-- Duck : lives in
    Duck --o Flock
    Customer "1" --> "*" Order : places
    Order "*" <-- "1" Customer2
    Client ..> Service : uses
    Service2 <.. Client2
    Duck ..|> Swimmer
    Swimmer2 <|.. Fish
    A -- B
    C .. D
    E <--> F
""")
    t = _edge_tuples(d)
    assert t == [
        ("Duck", "Animal", "inherit", ""), ("Fish", "Animal", "inherit", ""),
        ("Car", "Wheel", "compose", ""), ("Car2", "Wheel", "compose", ""),
        ("Pond", "Duck", "aggregate", "lives in"), ("Flock", "Duck", "aggregate", ""),
        ("Customer", "Order", "assoc_dir", "places"), ("Customer2", "Order", "assoc_dir", ""),
        ("Client", "Service", "depend", "uses"), ("Client2", "Service2", "depend", ""),
        ("Duck", "Swimmer", "realize", ""), ("Fish", "Swimmer2", "realize", ""),
        ("A", "B", "assoc", ""), ("C", "D", "assoc", ""), ("E", "F", "assoc", ""),
    ]
    ends = [(e.source_label, e.target_label) for e in d.edges]
    assert ends[2] == (None, "4")
    assert ends[3] == ("1", "1..*")  # reversed: the whole (Car2) is the source
    assert ends[6] == ("1", "*")
    assert ends[7] == ("1", "*")  # reversed arrow
    assert [e.style for e in d.edges][12:] == [None, "dashed", None]


def test_class_bodies_annotations_generics_labels():
    d = _parse("""classDiagram
    direction LR
    class Shape~T~ {
        <<interface>>
        +area() double
        +String name
    }
    class Color {
        <<enumeration>>
        RED
        GREEN
    }
    class Base
    <<abstract>> Base
    class Repo["User Repository"]
    Repo : +save(u User) bool
    Repo : -Map~K,V~ cache
    class Svc <<Service>>
    Square~T~ --|> Shape
""")
    assert d.direction == "LR"
    shape = _node(d, "Shape<T>")
    assert shape.kind == "interface"
    assert [(m.kind, m.text) for m in shape.members] == [("attr", "+String name"), ("op", "+area() : double")]
    color = _node(d, "Color")
    assert color.kind == "enum"
    assert [(m.kind, m.text) for m in color.members] == [("lit", "RED"), ("lit", "GREEN")]
    assert _node(d, "Base").kind == "abstract"
    repo = _node(d, "User Repository")
    assert [m.text for m in repo.members] == ["-Map<K,V> cache", "+save(u User) : bool"]
    svc = _node(d, "Svc")
    assert svc.kind == "class" and svc.stereotype == "Service"
    assert _edge_tuples(d) == [("Square<T>", "Shape<T>", "inherit", "")]


def test_class_namespaces_and_notes():
    d = _parse("""classDiagram
    note "General remark"
    namespace Shapes {
        class Triangle
        class Square {
            +side int
        }
    }
    note for Square "Can have\\nmany sides"
    Triangle --> Square
    Note --> Triangle
""")
    assert _edge_tuples(d)[-1] == ("Note", "Triangle", "assoc_dir", "")  # a class named Note
    ns = d.groups[0]
    assert (ns.kind, ns.text, ns.members) == ("namespace", "Shapes", _ids(d, "Triangle", "Square"))
    notes = [n for n in d.nodes if n.kind == "note"]
    assert [n.text for n in notes] == ["General remark", "Can have\nmany sides"]
    assert (d.edges[0].source, d.edges[0].target, d.edges[0].kind) == (notes[1].id, _node(d, "Square").id,
                                                                        "anchor")


@pytest.mark.parametrize("code", [
    "classDiagram\nclass A {\n+x",
    "classDiagram\nnamespace N {\nclass A",
    "classDiagram\nA ==> B",
])
def test_class_malformed(code):
    with pytest.raises(MermaidError):
        mermaid_to_diagram(code)


# --------------------------------------------------------------------------- #
# State diagrams
# --------------------------------------------------------------------------- #
def test_state_initial_final_per_scope_and_composites():
    d = _parse("""stateDiagram-v2
    [*] --> Idle
    Idle --> Active : start
    state Active {
        [*] --> Running
        Running --> Paused : pause
        Paused --> Running
        Running --> [*]
    }
    Active --> Idle : stop
    Idle --> [*]
    Idle --> [*] : quit
""")
    initials = [n for n in d.nodes if n.kind == "initial"]
    finals = [n for n in d.nodes if n.kind == "final"]
    assert len(initials) == 2 and len(finals) == 2  # top level + composite, shared within a scope
    comp = d.groups[0]
    assert (comp.kind, comp.text, comp.parent) == ("composite", "Active", None)
    assert set(comp.members) == {initials[1].id, finals[0].id, *_ids(d, "Running", "Paused")}
    idle = _node(d, "Idle").id
    assert ("Idle", "Active", "transition", "start") in _edge_tuples(d)
    assert any(e.source == comp.id and e.target == idle for e in d.edges)  # edges hit the group
    assert sum(e.target == finals[1].id for e in d.edges) == 2


def test_state_descriptions_aliases_and_pseudostates():
    d = _parse("""stateDiagram
    direction LR
    state "Waiting for input" as Wait
    state Check <<choice>>
    state F <<fork>>
    state J <<join>>
    [*] --> Wait
    Wait --> Check
    Check --> F : ok
    Check --> Done : bad
    F --> A
    F --> B
    A --> J
    B --> J
    J --> Done
    Done : All finished
    Done : exit code 0
    Late --> Late2
    state Late <<choice>>
""")
    assert d.direction == "LR"
    kinds = {n.text: n.kind for n in d.nodes if n.kind in ("state", "note")}
    assert kinds["Waiting for input"] == "state"
    assert sorted(n.kind for n in d.nodes if n.text == "") == ["choice", "fork", "initial", "join"]
    done = _node(d, "All finished")  # the first description replaces the name
    assert [m.text for m in done.members] == ["exit code 0"]
    # <<choice>> after the state was already used: Mermaid draws a plain state
    assert _node(d, "Late").kind == "state"


def test_state_concurrent_regions_and_notes():
    d = _parse("""stateDiagram-v2
    state Active {
        [*] --> NumLockOff
        NumLockOff --> NumLockOn : press
        --
        [*] --> CapsOff
        CapsOff --> CapsOn : press
    }
    note right of Active : two regions
    note left of NumLockOn
        line one
        line two
    end note
""")
    comp = next(g for g in d.groups if g.kind == "composite")
    regions = [g for g in d.groups if g.kind == "region"]
    assert len(regions) == 2 and all(r.parent == comp.id for r in regions)
    assert comp.members == []
    r1 = {n.text or n.kind for n in d.nodes if n.id in regions[0].members}
    r2 = {n.text or n.kind for n in d.nodes if n.id in regions[1].members}
    assert r1 == {"initial", "NumLockOff", "NumLockOn"} and r2 == {"initial", "CapsOff", "CapsOn"}
    notes = [n for n in d.nodes if n.kind == "note"]
    assert [n.text for n in notes] == ["two regions", "line one\nline two"]
    anchors = [(e.source, e.target) for e in d.edges if e.kind == "anchor"]
    assert anchors == [(notes[0].id, comp.id), (notes[1].id, _node(d, "NumLockOn").id)]


def test_state_stray_annotation_is_a_state_like_mermaid_draws_it():
    # Without `state`, Mermaid lexes <<fork>> as a separate state id (drawn "<>").
    d = _parse("stateDiagram-v2\nA --> F <<fork>>\nJ <<join>> --> B\nC --> <<fork>>")
    assert _edge_tuples(d) == [("A", "F", "transition", ""), ("<>", "B", "transition", ""),
                               ("C", "<>", "transition", "")]
    assert sorted(n.text for n in d.nodes) == ["<>", "<>", "A", "B", "C", "F", "J"]


@pytest.mark.parametrize("code", [
    "stateDiagram-v2\nstate A {\n[*] --> B",
    "stateDiagram-v2\n}",
    "stateDiagram-v2\n--",
    "stateDiagram-v2\nnote right of A\nnever closed",
])
def test_state_malformed(code):
    with pytest.raises(MermaidError):
        mermaid_to_diagram(code)


# --------------------------------------------------------------------------- #
# ER diagrams
# --------------------------------------------------------------------------- #
def test_er_cardinalities_attributes_and_aliases():
    d = _parse("""erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE-ITEM : contains
    CUSTOMER }|..|{ DELIVERY-ADDRESS : uses
    ORDER |o--o| INVOICE : "billed by"
    p[Person] one or more optionally to zero or one c["Car"] : drives
    CUSTOMER {
        string name PK "full name"
        string email UK
        int addr_id PK, FK
    }
    "Line Item"
""")
    rels = [(e.source_label, e.target_label, e.label, e.style, e.kind) for e in d.edges]
    assert rels == [
        ("1", "0..*", "places", None, "rel"),
        ("1", "1..*", "contains", None, "rel"),
        ("1..*", "1..*", "uses", "dashed", "rel"),
        ("0..1", "0..1", "billed by", None, "rel"),
        ("1..*", "0..1", "drives", "dashed", "rel"),
    ]
    cust = _node(d, "CUSTOMER")
    assert [m.text for m in cust.members] == [
        "string name PK full name", "string email UK", "int addr_id PK, FK"]
    assert all(m.kind == "attr" for m in cust.members)
    assert _ids(d, "Person", "Car") == [d.edges[4].source, d.edges[4].target]
    assert _node(d, "Line Item").kind == "entity"


def test_er_malformed():
    with pytest.raises(MermaidError):
        mermaid_to_diagram("erDiagram\nA {\n string x")


# --------------------------------------------------------------------------- #
# Mindmaps
# --------------------------------------------------------------------------- #
def test_mindmap_tree_and_shapes():
    d = _parse("""mindmap
  root((Main))
    A[Square]
      ::icon(fa fa-book)
      a1(Rounded)
      a2))Bang((
    B)Cloud(
      b1{{Hex}}
      :::urgent
    Plain text
""")
    assert [(n.kind, n.text) for n in d.nodes] == [
        ("circle", "Main"), ("box", "Square"), ("rounded", "Rounded"), ("other", "Bang"),
        ("cloud", "Cloud"), ("hexagon", "Hex"), ("node", "Plain text")]
    t = {(d.nodes[int(e.source[1:]) - 1].text, d.nodes[int(e.target[1:]) - 1].text) for e in d.edges}
    assert t == {("Main", "Square"), ("Square", "Rounded"), ("Square", "Bang"), ("Main", "Cloud"),
                 ("Cloud", "Hex"), ("Main", "Plain text")}
    assert all(e.kind == "line" for e in d.edges)


def test_mindmap_single_root():
    with pytest.raises(MermaidError):
        mermaid_to_diagram("mindmap\n  A\n  B")


# --------------------------------------------------------------------------- #
# Architecture (architecture-beta) and C4 -> graph
# --------------------------------------------------------------------------- #
def test_architecture_services_groups_edges():
    d = _parse("""architecture-beta
    group api(cloud)[API]
    group inner(server)[Inner] in api
    service db(database)[Database] in api
    service disk1(disk)[Storage] in inner
    service server(server)[Server] in inner
    service user(user)[User]
    junction j1
    db:L -- R:server
    disk1:T --> B:server
    server:R <-- L:user
    user{group}:B <--> T:j1
    server{group}:R --> L:user
""")
    assert [(n.kind, n.text) for n in d.nodes] == [
        ("cylinder", "Database"), ("storage", "Storage"), ("box3d", "Server"), ("other", "User"),
        ("point", "")]
    api, inner = d.groups
    assert (api.kind, api.text, api.parent, api.members) == ("cluster", "API", None, _ids(d, "Database"))
    assert (inner.text, inner.parent, inner.members) == ("Inner", api.id, _ids(d, "Storage", "Server"))
    db, disk, server, user, j1 = (n.id for n in d.nodes)
    assert [(e.source, e.target, e.kind) for e in d.edges] == [
        (db, server, "line"), (disk, server, "arrow"), (user, server, "arrow"),
        (user, j1, "bidir"),  # user has no group: the edge stays on the service
        (inner.id, user, "arrow")]


def test_c4_elements_boundaries_relations():
    d = _parse("""C4Container
    title Shop
    Person(cust, "Customer", "Buys things")
    Enterprise_Boundary(b0, "Shop Inc") {
        System_Boundary(b1, "Shop") {
            Container(web, "Web App", "React", "Sells things")
            ContainerDb(db, "Orders DB", "Postgres", "Stores orders")
            ContainerQueue_Ext(q, "Events")
        }
    }
    System_Ext(pay, "Payments", "Takes money", $tags="v1")
    UpdateRelStyle(cust, web, $offsetY="-30")
    Rel(cust, web, "Uses", "HTTPS")
    BiRel(web, db, "Reads/writes")
    Rel_Back(pay, web, "Charges")
    Rel_D(web, b1, "Lives in")
""")
    assert d.title == "Shop"
    assert [(n.kind, n.stereotype, n.text) for n in d.nodes] == [
        ("actor", "person", "Customer\nBuys things"),
        ("box", "container", "Web App\n[React]\nSells things"),
        ("cylinder", "container_db", "Orders DB\n[Postgres]\nStores orders"),
        ("queue", "external_container_queue", "Events"),
        ("box", "external_system", "Payments\nTakes money"),
    ]
    b0, b1 = d.groups
    assert (b0.text, b0.parent, b0.members) == ("Shop Inc\n[ENTERPRISE]", None, [])
    cust, web, db, q, pay = (n.id for n in d.nodes)
    assert (b1.text, b1.parent, b1.members) == ("Shop\n[SYSTEM]", b0.id, [web, db, q])
    assert [(e.source, e.target, e.kind, e.label) for e in d.edges] == [
        (cust, web, "arrow", "Uses\n[HTTPS]"), (web, db, "bidir", "Reads/writes"),
        (web, pay, "arrow", "Charges"), (web, b1.id, "arrow", "Lives in")]


@pytest.mark.parametrize("code", [
    "architecture-beta\nservice a(server)[A]\nend",
    "architecture-beta\nservice a(server)[A] in nowhere",
    "architecture-beta\nservice a(server)[A]\na:R --> L:b",
    "C4Context\nPerson(a, \"A\")\nRel(a, b, \"x\")",
    "C4Context\nSystem_Boundary(b, \"B\") {\nPerson(a, \"A\")",
    "C4Context\nFoo(a, \"A\")",
])
def test_architecture_and_c4_malformed(code):
    with pytest.raises(MermaidError):
        mermaid_to_diagram(code)


def test_outputs_serialize_canonically():
    for code in ("sequenceDiagram\nA->>B: x\nloop l\nB-->>A: y\nend",
                 "classDiagram\nA <|-- B", "stateDiagram-v2\n[*] --> A\nA --> [*]",
                 "erDiagram\nA ||--o{ B : r", "mindmap\n  R\n    C"):
        c = canonicalize(mermaid_to_diagram(code))
        assert dumps(canonicalize(c)) == dumps(c)
