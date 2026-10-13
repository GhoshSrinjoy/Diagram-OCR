# Diagram IR — specification v1.0

Diagram IR is the single structured representation for every diagram this project
handles: flowcharts, UML activity / sequence / class / state diagrams, ER diagrams,
logic and electrical circuits, and generic node-link graphs.

There is **one data model** and **four serializations** of it:

| Format | Used for | Lossless |
|---|---|---|
| **DGM** (compact text, `.dgm`) | What the model generates. Training targets. | yes |
| **JSON** (`.json`) | Reference interchange format; checked by `schema.json` | yes |
| **YAML** (`.yaml`) | Same object as JSON, easier to read and edit | yes |
| **XML** (`.xml`) | Element-for-element mapping of the JSON object | yes |

Any format converts to any other through the data model (`python -m diagram_ir convert`).

Design rules:

1. **The model writes one compact format.** JSON, YAML and XML are produced by code,
   never by the model. DGM uses 3–5× fewer tokens than pretty JSON and about 2× fewer
   than minified JSON (measured with the Qwen tokenizer on `examples/`), so generation
   is faster and the output can be grammar-constrained.
2. **Layout-aware but layout-optional.** Every element may carry a box in normalized
   0–1000 coordinates (the Qwen-VL grounding convention). Coordinates are about half
   the output tokens; leave them out (structure-only mode) when locations are not
   needed.
3. **Canonical form.** Each diagram has exactly one canonical serialization (stable ids,
   flow-ordered nodes). Training targets are always canonical.
4. **Closed vocabularies per diagram type**, with `note` / `text` / `other` as escape hatches.

---

## 1. Data model

### Diagram

| Field | Type | Notes |
|---|---|---|
| `type` | enum | `flowchart`, `activity`, `sequence`, `class`, `state`, `er`, `circuit`, `graph` |
| `direction` | enum? | Main layout direction: `TB`, `BT`, `LR`, `RL` |
| `title` | string? | Diagram title, if drawn |
| `nodes` | Node[] | |
| `edges` | Edge[] | Order matters for `sequence` (time order) |
| `groups` | Group[] | Containers |
| `attrs` | map<string,string> | Extension point |

### Node

| Field | Type | Notes |
|---|---|---|
| `id` | id | Canonical: `n1`, `n2`, … |
| `kind` | kind | Shape or semantic class, from the type's vocabulary (§2) |
| `text` | string | Visible text. `""` if none. Explicit line breaks are kept as `\n`. |
| `bbox` | [x1,y1,x2,y2]? | Normalized 0–1000 |
| `stereotype` | string? | UML `«…»` text, without the guillemets |
| `members` | Member[] | Compartment lines (classes, entities, objects, states) |
| `attrs` | map | Extension point |

**Member**: `kind` ∈ `attr` (attribute / field / column), `op` (operation / method),
`lit` (enum literal); `text` is the line exactly as drawn (`-id: int`,
`+total(): float`, `id PK int`); optional `bbox`.

### Edge

| Field | Type | Notes |
|---|---|---|
| `id` | `e<N>` | **Positional**: the i-th edge is `e{i}`. Not stored, derived. |
| `source`, `target` | id | Node id **or group id** (e.g. a transition into a composite state) |
| `kind` | kind | From the type's edge vocabulary. Each type has a default (§2). |
| `directed` | bool | Derived from `kind` (JSON/XML output only) |
| `label` | string? | Text on the line: condition, message, guard, event |
| `source_label`, `target_label` | string? | End labels: multiplicities, roles, cardinalities |
| `source_port`, `target_port` | string? | Pin names / indices (circuits, non-commutative inputs) |
| `style` | enum? | `dashed`, `dotted`, `thick`; absent = solid |
| `path` | [[x,y],…]? | Polyline from source to target, ≥ 2 points |
| `attrs` | map | Extension point |

Direction conventions:
- `inherit` / `realize`: source = subclass / implementer, target = parent / interface.
- `compose` / `aggregate`: source = the whole (diamond end), target = the part.
- `depend`: source = client, target = supplier.
- Digital `wire`: source = driver (output pin), target = sink (input pin). Fan-out is
  one edge per sink; junction dots are resolved, not stored, unless they matter (`junction` node).

### Group

| Field | Type | Notes |
|---|---|---|
| `id` | id | Canonical: `g1`, `g2`, … |
| `kind` | kind | lane, subgraph, package, composite, fragment kind, … |
| `text` | string | Group title / guard |
| `members` | id[] | Direct members: node ids and/or edge ids (sequence fragments hold messages) |
| `parent` | id? | Enclosing group. Nesting is expressed **only** via `parent`, never by listing a group in `members`. |
| `bbox` | box? | |

A sequence-diagram `alt` with branches is a group of kind `alt`, plus one `operand`
child group per branch (`parent=` the alt) whose `text` is the guard and whose members
are the messages in that branch. See `examples/sequence.dgm`.

### Coordinates

Integers in `0..1000`, normalized to the image: `x = round(1000 * px / width)`,
`y = round(1000 * py / height)`. `bbox = [x1, y1, x2, y2]` with `x1 ≤ x2`, `y1 ≤ y2`.

### Ids

`[A-Za-z_][A-Za-z0-9_]*`, unique across nodes and groups. Ids of the form `e<N>` are
reserved for edges. Non-canonical ids (e.g. Mermaid's `A`, `B`) are accepted on input;
canonicalization renames them.

---

## 2. Vocabularies

Every type also accepts node kinds `note`, `text` (free-floating text), `other`, edge
kind `anchor` (undirected note attachment), and group kind `group`.
**Bold** = default edge kind (omitted in DGM). *Italic* edge kinds are undirected.

| Type | Node kinds | Edge kinds | Group kinds |
|---|---|---|---|
| `flowchart` | terminal, process, decision, io, subroutine, database, document, manual_input, manual_op, preparation, connector, offpage, delay, display, loop_limit | **flow**, *line*, bidir | subgraph, lane |
| `activity` | action, initial, final, flow_final, decision, merge, fork, join, object, send_signal, receive_signal, time_event | **flow** | lane, region |
| `sequence` | participant, actor, boundary, control, entity, database, collections, queue | **sync**, async, reply, create, destroy | alt, opt, loop, par, break, critical, neg, ref, operand, box |
| `class` | class, abstract, interface, enum | ***assoc***, assoc_dir, inherit, realize, compose, aggregate, depend | package, namespace |
| `state` | state, initial, final, choice, fork, join, history, deep_history | **transition** | composite, region |
| `er` | entity, weak_entity, relationship, attribute | ***rel*** | — |
| `circuit` | input, output, and, or, not, nand, nor, xor, xnor, buf, mux, demux, dff, latch, const, clock, junction, resistor, capacitor, inductor, diode, led, transistor, opamp, voltage_source, current_source, battery, ground, switch, lamp, ic, port | **wire**, *conn* | block |
| `graph` | node, box, rounded, ellipse, circle, diamond, parallelogram, hexagon, cylinder, point, image, box3d, cloud, folder, component, actor, document, queue, storage, file | **arrow**, *line*, bidir | cluster |

Kinds describe **what is drawn**. A rectangle reading "Start" is a `process`; only a
rounded/stadium terminator is a `terminal`. Flowchart kinds map 1:1 to the standard
ISO 5807 shapes. `circuit` uses `wire` (directed signal flow) for digital logic and
`conn` (undirected) for electrical schematics.

---

## 3. DGM — compact text format

UTF-8, one element per line. Blank lines and lines starting with `#` are ignored.

```
header := "diagram" SP TYPE (SP attr)*                         # first non-comment line
node   := "node" SP ID SP KIND [SP STRING] [SP BBOX] (SP attr)*
member := INDENT? ("attr"|"op"|"lit") SP STRING [SP BBOX]      # attaches to previous node
edge   := "edge" SP ID SP ID [SP KIND] [SP STRING] (SP attr)*  # KIND omitted = type default
group  := "group" SP ID SP KIND [SP STRING] [SP BBOX] (SP attr)* [SP ":" (SP ID)+]

BBOX   := "@" INT "," INT "," INT "," INT
attr   := KEY "=" (BARE | STRING)
BARE   := [A-Za-z0-9_.*+\-,]+                                  # anything else is quoted
STRING := '"' (char | "\\" | '\"' | "\n" | "\t")* '"'
```

Attribute keys:

| Line | Key → field |
|---|---|
| header | `dir` → direction, `title` → title |
| node | `stereo` → stereotype |
| edge | `sl`/`tl` → source/target_label, `sp`/`tp` → source/target_port, `style`, `path=x,y,x,y,…` |
| group | `parent` |

Any other `key=value` goes to `attrs`. Canonical line order: header, nodes (each followed
by its members), edges, groups. The parser accepts any order.

Example (`examples/flowchart.dgm`):

```
diagram flowchart dir=TB
node n1 terminal "Start" @400,20,600,70
node n2 process "Install Engine" @380,110,620,160
node n6 decision "Quality Check" @420,470,580,570
node n7 process "Drivetrain Installation Complete" @150,640,450,690
node n8 process "Review & Correct Issues" @600,640,850,690
edge n5 n6
edge n6 n7 "Pass"
edge n6 n8 "Fail"
edge n8 n2 style=dashed
```

Class diagram (`examples/class.dgm`, excerpt):

```
diagram class
node n1 class "Customer" @40,40,260,200
  attr "-name: String"
  op "+placeOrder(): Order"
node n2 class "Order" @380,40,620,220 stereo=entity
edge n1 n2 "places" sl=1 tl=0..*
edge n2 n3 realize
edge n2 n5 compose sl=1 tl=1..*
```

**Lenient parsing.** `diagram_ir.parse()` skips malformed lines and returns them as
errors, so a generation cut off mid-line still yields every complete element. A missing
header falls back to type `graph`. Use `loads()` for strict parsing.

---

## 4. JSON, YAML, XML

### JSON / YAML

The JSON object is the reference shape. Required keys are always present, even when
empty (`nodes`, `edges`, `groups`, node `text`, group `members`). Optional fields
are omitted when empty. `directed` is output-only; on input it is ignored, and a missing
edge `kind` takes the type's default. Edge `id`s on input may be arbitrary; they are
renumbered positionally and group `members` are remapped. Full schema: `schema.json`.

```json
{
  "format": "diagram-ir",
  "version": "1.0",
  "type": "flowchart",
  "direction": "TB",
  "nodes": [
    {"id": "n1", "kind": "terminal", "text": "Start", "bbox": [400, 20, 600, 70]},
    {"id": "n2", "kind": "decision", "text": "Stock OK?", "bbox": [430, 120, 570, 220]},
    {"id": "n3", "kind": "process", "text": "Ship order", "bbox": [150, 280, 400, 330]},
    {"id": "n4", "kind": "process", "text": "Backorder", "bbox": [600, 280, 850, 330]}
  ],
  "edges": [
    {"id": "e1", "source": "n1", "target": "n2", "kind": "flow", "directed": true},
    {"id": "e2", "source": "n2", "target": "n3", "kind": "flow", "directed": true,
     "label": "yes"},
    {"id": "e3", "source": "n2", "target": "n4", "kind": "flow", "directed": true,
     "label": "no", "style": "dashed"}
  ],
  "groups": [
    {"id": "g1", "kind": "lane", "text": "Warehouse", "members": ["n1", "n2", "n3", "n4"],
     "bbox": [0, 0, 1000, 400]}
  ]
}
```

(The converter prints one field per line; it is condensed here.) YAML is the same object.
Ambiguous scalars such as `yes`, `no`, `1` and `null` are quoted automatically, and
multi-line text uses `|` blocks.

### XML

A fragment showing each construct (not one complete diagram):

```xml
<diagram format="diagram-ir" version="1.0" type="flowchart" direction="TB">
  <nodes>
    <node id="n1" kind="terminal" bbox="400,20,600,70"><text>Start</text></node>
    <node id="n5" kind="class" stereotype="entity">
      <text>Order</text>
      <member kind="attr">-id: int</member>
    </node>
  </nodes>
  <edges>
    <edge id="e2" source="n2" target="n3" kind="flow" directed="true"><label>yes</label></edge>
    <edge id="e4" source="n5" target="n6" kind="compose" directed="true"
          source_label="1" target_label="1..*"/>
    <edge id="e5" source="n1" target="n2" kind="flow" directed="true" path="10,20 10,80 90,80"/>
  </edges>
  <groups>
    <group id="g1" kind="lane" bbox="0,0,1000,400"><text>Warehouse</text><ref id="n1"/></group>
  </groups>
</diagram>
```

Rules:
- Scalars are XML attributes with the JSON names.
- Free text goes in child elements (`<title>`, `<text>`, `<label>`, `<member>`) so line
  breaks survive.
- `bbox` is `"x1,y1,x2,y2"` and `path` is `"x,y x,y …"`.
- Extension attrs are `<attr name="k">v</attr>`.
- `<text>` / `<label>` are omitted when empty.

---

## 5. Canonical form (training targets)

`canonicalize()` produces the one serialization used as the training target:

1. **Text:** whitespace runs collapse to one space, each line is trimmed, and leading
   and trailing blank lines are dropped. Explicit line breaks are kept.
2. **Node order:**
   - *Flow types* (`flowchart`, `activity`, `state`, `graph`, `circuit`): follow the
     arrows. Entry nodes are nodes with no incoming one-way arrow, excluding notes. Each
     connected piece of the diagram is walked in turn (the piece holding the best entry node
     first), with a BFS that starts from all of that piece's entry nodes at once. Kinds `terminal` / `initial` / `input` come first,
     then reading order. Branches are visited in reading order when coordinates exist,
     otherwise in the order the edges were written. Arrows into a group count as arrows
     into its first member. Unreached nodes (entry-less cycles, notes) continue the BFS
     in reading order.
   - *Layout types* (`class`, `er`, `sequence`): reading order. Boxes share a row when
     they overlap vertically by at least 50% of the smaller box; rows run top to bottom,
     left to right within a row.
   - Without coordinates on every node, the original order breaks ties.
3. **Ids:** nodes become `n1…nN` in that order. Groups become `g1…gM`, ordered by their
   earliest member (descendants included), parents before children.
4. **Edge order:** undirected and double-headed edges are first stored from the earlier
   endpoint to the later one (end labels and ports swap with them). Edges are then sorted by
   (source, target, kind, label); a container endpoint sorts right after the node where the
   flow enters it (its `initial` nodes, else members with no incoming arrow from inside, else
   all members). In `sequence` diagrams the written order is time order and is kept.
5. **Group members:** nodes in node order, then edges in edge order, deduplicated.

`canonicalize` is idempotent on all converted training data. A known residual: on about
0.1% of real-world PlantUML graphs with undirected lines into containers, a second pass
can still reorder nodes. Metrics are order-independent, so this does not affect scoring. Following the flow means the model emits each node right
after the node that points to it, the "visual causal flow" idea from DeepSeek-OCR 2.

---

## 6. Validation

`validate(d, strict=False)` returns a list of issues:

- **Errors** (structural):
  - unknown type or direction
  - malformed or duplicate ids, or ids of the form `e<N>`
  - edges or members pointing to missing elements
  - group listed as a member
  - parent cycles
  - bbox or path outside 0–1000 or inverted
  - bad member kind or style
- **Warnings**:
  - kinds outside the type's vocabulary (these are errors with `strict=True`)
  - members on a kind that normally has none
  - duplicate group members

---

## 7. Command line

```
cd src
python -m diagram_ir convert diagram_ir/examples/flowchart.dgm --to json
python -m diagram_ir convert model_output.dgm --to json,yaml,xml -o exports/
python -m diagram_ir convert some.json --to dgm --canonical
python -m diagram_ir validate exports/flowchart.xml --strict
```

## 8. Versioning

`version` is `MAJOR.MINOR`. Adding kinds, types or optional fields bumps MINOR.
Readers must accept unknown `attrs`. Changing the meaning of an existing field bumps
MAJOR, and readers reject unknown majors. Candidates for 1.1: `bpmn`, `usecase` and
`component` types, edge-label boxes.
