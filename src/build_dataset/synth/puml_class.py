"""Synthetic PlantUML class diagrams: 2-14 classes / abstract classes / interfaces / enums with typed
members in one of several language flavours, stereotypes, generics, packages and namespaces, every
relation kind with multiplicities and labels, and notes. `generate` returns the PlantUML body and
the Diagram the importer should produce for it (see puml_check)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from diagram_ir import Diagram

from .puml_check import Intent
from .puml_words import (
    CLASS_STEREOTYPES,
    GENERIC_FIELDS,
    GENERIC_LITERALS,
    GENERIC_VERBS,
    TYPES,
    Domain,
    camel,
    is_keyword,
    pascal,
    snake,
    words,
)

FLAVORS = ("java", "java", "cs", "ts", "py", "kotlin", "plain")
SUFFIXES = ("Service", "Repository", "Controller", "Manager", "Factory", "Validator", "Mapper", "Handler",
            "Dto", "Builder", "Client", "Policy", "Config", "View", "Record", "Event")
ABLE = ("Payable", "Shippable", "Trackable", "Auditable", "Searchable", "Exportable", "Notifiable",
        "Billable", "Schedulable", "Comparable", "Serializable", "Cacheable", "Identifiable", "Versioned",
        "Printable", "Observable")
MULTS = ("1", "0..1", "0..*", "1..*", "*", "many", "n", "1..n", "2..*", "0..n", "3")
LINK_LABELS = ("places", "contains", "owns", "uses", "manages", "has", "belongs to", "references", "creates",
               "notifies", "schedules", "assigned to", "tracks", "pays", "issues", "holds", "produces",
               "validates", "reads", "updates")
NOTE_TEXTS = ("Aggregate root", "Immutable value object", "Persisted via JPA", "Uses optimistic locking",
              "Deprecated since v2", "See ADR-12", "Cached for 5 minutes", "Thread-safe", "Validated on save",
              "Soft delete only", "Created by the factory", "One per tenant", "Audited", "Read-only view",
              "Implements retry with backoff", "Lazy loaded", "Exposed via REST", "Singleton instance",
              "Encrypted at rest", "Max 50 items")
MODULES = ("domain", "model", "service", "services", "repository", "api", "web", "core", "infrastructure",
           "persistence", "billing", "orders", "accounts", "catalog", "shipping", "common", "util", "events",
           "security", "reporting", "integration", "admin", "app", "data")
QUOTED_MODULES = ("Order Management", "Payment Processing", "Core Domain", "Data Access", "User Interface",
                  "Shared Kernel", "Customer Accounts", "Back Office", "Public API", "Domain Model")
# html tags PlantUML would read inside `<...>` (a generic argument must not be one of them)
_TAGS = frozenset("b i u s w del strike color size font back img sub sup code plain em strong tt big small"
                  .split())


@dataclass
class _Cls:
    name: str  # declared name
    kind: str  # class | abstract | interface | enum
    key: str = ""  # how links refer to it (qualified inside a namespace)
    generic: str = ""
    display: Optional[str] = None
    stereos: list[str] = field(default_factory=list)
    body: list[str] = field(default_factory=list)  # member lines as written
    members: list[tuple[str, str]] = field(default_factory=list)  # (member kind, drawn text)
    pkg: Optional[int] = None
    nid: str = ""
    style: str = "body"  # body | lines | bare | implicit
    literals: tuple[str, ...] = ()  # enum literal pool

    @property
    def text(self) -> str:
        return (self.display or self.name) + self.generic


class _Gen:
    def __init__(self, rng: random.Random, dom: Domain):
        self.rng = rng
        self.dom = dom
        self.flavor = rng.choice(FLAVORS)
        self.prefix_types = self.flavor == "java" and rng.random() < 0.3  # `-String name`
        self.vis = rng.random() < (0.5 if self.flavor == "py" else 0.85)
        self.spaced_colon = rng.random() < 0.25
        self.taken: set[str] = set()

    # ---- names ---------------------------------------------------------------
    def fresh(self, cands: list[str]) -> Optional[str]:
        for c in cands:
            if c.lower() not in self.taken and not is_keyword(c) and c.isidentifier():
                self.taken.add(c.lower())
                return c
        return None

    def class_name(self) -> str:
        ents = list(self.dom.entities)
        self.rng.shuffle(ents)
        name = self.fresh(ents)
        if name is None:
            combos = [e + s for e in ents for s in SUFFIXES]
            self.rng.shuffle(combos)
            name = self.fresh(combos)
        return name or self.fresh([f"Class{k}" for k in range(1, 99)]) or "Thing"

    def abstract_name(self) -> str:
        ents = list(self.dom.entities)
        self.rng.shuffle(ents)
        if self.rng.random() < 0.5:
            pre = self.rng.choice(["Abstract", "Base"])
            got = self.fresh([pre + e for e in ents])
            if got:
                return got
        return self.class_name()

    def interface_name(self) -> str:
        ents = list(self.dom.entities)
        self.rng.shuffle(ents)
        r = self.rng.random()
        if r < 0.3:
            cands = list(ABLE)
            self.rng.shuffle(cands)
        elif r < 0.5 or self.flavor == "cs":
            cands = ["I" + e for e in ents]
        else:
            suffix = self.rng.choice(["Repository", "Service", "Gateway", "Listener", "Provider", "Port"])
            cands = [e + suffix for e in ents]
        return self.fresh(cands) or self.class_name()

    def enum_name(self) -> tuple[str, tuple[str, ...]]:
        enums = list(self.dom.enums)
        self.rng.shuffle(enums)
        for name, lits in enums:
            if self.fresh([name]):
                return name, lits
        lits = self.rng.choice(GENERIC_LITERALS)
        name = self.fresh(["Priority", "Level", "Mode", "Category", "Kind", "Grade", "Region"])
        return name or "Flag", lits

    # ---- members ---------------------------------------------------------------
    def ident(self, ws: list[str]) -> str:
        if self.flavor == "cs":
            return pascal(ws)
        return snake(ws) if self.flavor == "py" else camel(ws)

    def type_of(self, field_name: str, others: list[_Cls]) -> str:
        t = TYPES["java" if self.flavor == "plain" else self.flavor]
        n = field_name.lower().replace("_", "")
        if n.endswith("id") or n in ("iban", "isbn", "msisdn", "imsi", "iccid"):
            key = self.rng.choice(["long", "uuid", "int", "str"]) if n.endswith("id") else "str"
        elif n.endswith(("at", "time", "seen", "login")) or n in ("timestamp", "eta"):
            key = "dt"
        elif "date" in n or n in ("expiry", "dateofbirth", "duedate"):
            key = "date"
        elif n.startswith(("is", "has")) or n in ("enabled", "available", "roaming", "active", "isgift"):
            key = "bool"
        elif any(k in n for k in ("amount", "price", "total", "balance", "fee", "salary", "rate", "bonus",
                                  "discount", "premium", "loss", "limit", "value", "cost", "tip")):
            key = self.rng.choice(["dec", "dec", "float"])
        elif any(k in n for k in ("count", "quantity", "qty", "level", "seats", "capacity", "credits",
                                  "pages", "nights", "attempts", "minutes", "score", "rating", "km", "kg",
                                  "weight", "volume", "duration", "renewals", "years", "size", "guests",
                                  "points")):
            key = self.rng.choice(["int", "int", "float"])
        else:
            key = "str"
        return t[key]

    def field_line(self, others: list[_Cls], vis_weights=(50, 30, 13, 7)) -> tuple[str, str]:
        r = self.rng.random()
        if r < 0.55:
            name = self.rng.choice(self.dom.fields)
        elif r < 0.82 or not others:
            name = self.rng.choice(GENERIC_FIELDS)
        else:  # a reference to another class: `customer: Customer`, `items: List<OrderItem>`
            other = self.rng.choice(others)
            many = self.rng.random() < 0.4 and other.name.lower() not in _TAGS
            ws = words(other.name)
            name = camel(ws[:-1] + [ws[-1] + "s"]) if many else camel(ws)
        fname = self.ident(words(name))
        vis = self.rng.choices("-+#~", weights=vis_weights)[0] if self.vis else ""
        if self.flavor == "plain":
            return "attr", f"{vis}{fname}"
        if r >= 0.82 and others:
            t = TYPES["java" if self.flavor == "plain" else self.flavor]
            typ = other.name
            if many:
                typ = t[self.rng.choice(["list", "list", "set"])].replace("{}", other.name)
        else:
            typ = self.type_of(name, others)
        if self.prefix_types:
            return "attr", f"{vis}{typ} {fname}"
        sep = " : " if self.spaced_colon else ": "
        return "attr", f"{vis}{fname}{sep}{typ}"

    def param(self, pname: str, others: list[_Cls]) -> str:
        ptype, pid = self.type_of(pname, others), self.ident(words(pname))
        return f"{ptype} {pid}" if self.prefix_types else f"{pid}: {ptype}"

    def op_line(self, owner: _Cls, others: list[_Cls]) -> tuple[str, str]:
        rng = self.rng
        t = TYPES["java" if self.flavor == "plain" else self.flavor]
        fields = list(self.dom.fields) + list(GENERIC_FIELDS)
        params: list[str] = []
        r = rng.random()
        if r < 0.3:  # domain verb: ship(), refund(amount)
            ws = words(rng.choice(self.dom.verbs))
            if rng.random() < 0.5:
                params.append(self.param(rng.choice(fields), others))
            ret = t["void"] if rng.random() < 0.5 else t["bool"]
        elif r < 0.55:  # verb + entity: placeOrder(customerId), addItem(item: OrderItem)
            target = rng.choice([owner] * 2 + others[:4])
            ws = words(rng.choice(self.dom.verbs)) + words(target.name)[-1:]
            if target is not owner and rng.random() < 0.5 and target.name.lower() not in _TAGS:
                pid = self.ident(words(target.name)[-1:])
                params.append(f"{target.name} {pid}" if self.prefix_types else f"{pid}: {target.name}")
            elif rng.random() < 0.4:
                params.append(self.param(rng.choice(fields), others))
            ret = rng.choice([t["void"], t["bool"], target.name if target.kind != "enum" else t["void"]])
        elif r < 0.85:  # accessors: getTotalAmount(), setStatus(status), isActive()
            f = rng.choice(fields)
            pre = rng.choices(["get", "set", "update", "calculate"], weights=(45, 25, 15, 15))[0]
            ws = words(f) if words(f)[0] in ("is", "has") else [pre] + words(f)
            if ws[0] in ("is", "has"):
                pre = "is"
            if pre in ("set", "update"):
                params.append(self.param(f, others))
                ret = t["void"]
            else:
                ret = t["bool"] if pre in ("is", "has") else self.type_of(f, others)
        else:
            ws = words(rng.choice(GENERIC_VERBS))
            if rng.random() < 0.3:
                params.append(self.param(rng.choice(fields), others))
            ret = t["bool"] if ws[0] in ("is", "has", "check", "validate") else t["void"]
        if len(params) == 1 and rng.random() < 0.15:
            params.append(self.param(rng.choice(fields), others))
        name = self.ident(ws)
        vis = rng.choices("+-#~", weights=(65, 20, 10, 5))[0] if self.vis else ""
        if self.flavor == "plain":
            return "op", f"{vis}{name}()"
        sig = f"{name}({', '.join(params)})"
        if self.prefix_types:
            return "op", f"{vis}{ret} {sig}"
        if ret == t["void"] and rng.random() < 0.5:
            return "op", f"{vis}{sig}"
        sep = " : " if self.spaced_colon else ": "
        return "op", f"{vis}{sig}{sep}{ret}"

    def fill(self, c: _Cls, others: list[_Cls], budget: int) -> None:
        rng = self.rng
        lines: list[tuple[str, str, str]] = []  # (kind, drawn, modifier)
        if c.kind == "enum":
            lits = c.literals
            k = min(len(lits), rng.randint(2, 6))
            start = rng.randint(0, len(lits) - k)
            lines += [("lit", x, "") for x in lits[start:start + k]]
            if rng.random() < 0.12:  # an enum field: needs a visibility or a type to read as one
                vis = "-" if self.vis else ""
                flavor = "java" if self.flavor == "plain" else self.flavor
                lines.append(("attr", f"{vis}{self.ident(['code'])}: {TYPES[flavor]['int']}", ""))
        else:
            if c.kind == "interface":
                n_attr, n_op = (0 if rng.random() < 0.9 else 1), rng.randint(1, 5)
            elif c.kind == "abstract":
                n_attr, n_op = rng.randint(0, 4), rng.randint(1, 5)
            else:
                n_attr = rng.choices(range(7), weights=(12, 15, 20, 20, 15, 10, 8))[0]
                n_op = rng.choices(range(6), weights=(20, 22, 22, 18, 10, 8))[0]
            n_attr, n_op = min(n_attr, budget), min(n_op, max(0, budget - n_attr))
            seen = set()
            for _ in range(n_attr):
                kind, text = self.field_line(others)
                if text in seen:
                    continue
                seen.add(text)
                mod = "{static} " if rng.random() < 0.06 else ""
                lines.append((kind, text, mod))
            for _ in range(n_op):
                kind, text = self.op_line(c, others)
                if text in seen:
                    continue
                seen.add(text)
                abstract = c.kind in ("abstract", "interface") and rng.random() < 0.3
                mod = "{abstract} " if abstract else ("{static} " if rng.random() < 0.05 else "")
                lines.append((kind, text, mod))
        c.members = [(k, t) for k, t, _m in lines]
        c.body = [f"{m}{t}" for _k, t, m in lines]
        if c.kind != "enum" and len(lines) >= 3 and rng.random() < 0.1:  # a separator line
            pos = rng.randint(1, len(lines) - 1)
            c.body.insert(pos, rng.choice(["--", "..", "==", "__"]))


def _size(rng: random.Random) -> int:
    return rng.choices(range(2, 15), weights=(4, 7, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2))[0]


def generate(rng: random.Random, dom: Domain, direction: str = "TB",
             colors: bool = True) -> tuple[list[str], Diagram]:
    g = _Gen(rng, dom)
    n = _size(rng)
    n_enum = rng.choices([0, 1, 2], weights=(55, 35, 10))[0] if n >= 3 else 0
    n_iface = rng.choices([0, 1, 2, 3], weights=(45, 35, 15, 5))[0] if n >= 3 else rng.choice([0, 0, 1])
    n_abs = rng.choices([0, 1, 2], weights=(60, 30, 10))[0] if n >= 3 else 0
    while n_enum + n_iface + n_abs > n - 1:
        n_enum, n_iface, n_abs = max(0, n_enum - 1), max(0, n_iface - (n_enum == 0)), max(0, n_abs - 1)
    kinds = ["enum"] * n_enum + ["interface"] * n_iface + ["abstract"] * n_abs
    kinds += ["class"] * (n - len(kinds))
    rng.shuffle(kinds)

    classes: list[_Cls] = []
    for kind in kinds:
        if kind == "enum":
            name, lits = g.enum_name()
            c = _Cls(name, kind, literals=lits)
        elif kind == "interface":
            c = _Cls(g.interface_name(), kind)
        elif kind == "abstract":
            c = _Cls(g.abstract_name(), kind)
        else:
            c = _Cls(g.class_name(), kind)
        classes.append(c)

    budget_total = max(8, 54 - 2 * n)  # keeps big diagrams' targets short
    for c in classes:
        others = [o for o in classes if o is not c and o.kind != "enum"]
        c_budget = max(1, budget_total // n + rng.randint(0, 3))
        g.fill(c, others, c_budget)
        p_st = {"class": 0.15, "abstract": 0.1, "interface": 0.1, "enum": 0.05}[c.kind]
        if rng.random() < p_st:
            c.stereos = rng.sample(CLASS_STEREOTYPES, 2 if rng.random() < 0.08 else 1)
        if c.kind in ("class", "abstract", "interface") and rng.random() < 0.1:
            c.generic = rng.choice(["<T>", "<T>", "<K,V>", "<ID>", "<TItem>", "<E>", "<R>"])
        if c.kind == "class" and not c.generic and rng.random() < 0.05:
            ws = words(c.name)
            if len(ws) > 1:
                c.display = " ".join(w.capitalize() for w in ws)
        if c.kind == "class" and not c.members and not c.stereos and not c.generic and not c.display:
            c.style = "implicit" if rng.random() < 0.4 else "bare"
        elif c.kind in ("class", "abstract") and not c.generic and c.members and rng.random() < 0.12:
            c.style = "lines"

    # ---- packages ---------------------------------------------------------------
    packages: list[dict] = []  # {name, quoted, ns, parent, members}
    movable = [c for c in classes if c.style != "implicit"]
    rng.shuffle(movable)
    keep_top = 0 if n < 4 else rng.randint(0, max(0, len(movable) // 3))
    if n >= 3 and len(movable) - keep_top >= 1 and rng.random() < 0.35:
        k = rng.randint(1, max(1, min(3, n // 2, len(movable) - keep_top)))
        names: set[str] = set()
        for i in range(k):
            ns = rng.random() < 0.2
            quoted = not ns and rng.random() < 0.3
            pool = QUOTED_MODULES if quoted else MODULES
            cands = [x for x in pool
                     if x.lower() not in names and x.lower() not in g.taken and not is_keyword(x)]
            name = rng.choice(cands)
            names.add(name.lower())
            parent = 0 if (i > 0 and not ns and not packages[0]["ns"] and rng.random() < 0.15) else None
            packages.append({"name": name, "quoted": quoted, "ns": ns, "parent": parent, "members": []})
        for idx, c in enumerate(movable[keep_top:]):  # every package gets at least one class
            pi = idx if idx < len(packages) else rng.randrange(len(packages))
            c.pkg = pi
            packages[pi]["members"].append(c)
    for c in classes:
        c.key = c.name
        if c.pkg is not None and packages and packages[c.pkg]["ns"]:
            c.key = f"{packages[c.pkg]['name']}.{c.name}"
        if c.style == "lines" and c.pkg is not None:
            c.style = "body"

    # ---- intent: nodes and groups ------------------------------------------------------
    it = Intent("class", direction)
    for c in classes:
        c.nid = it.node(c.kind, c.text, ", ".join(c.stereos) or None, c.members)
    gids = []
    for p in packages:
        parent = gids[p["parent"]] if p["parent"] is not None else None
        grp = it.group("namespace" if p["ns"] else "package", p["name"], parent)
        grp.members = [c.nid for c in p["members"]]
        gids.append(grp.id)

    # ---- relations ---------------------------------------------------------------
    rels = _relations(rng, classes)
    link_lines = []
    for kind, s, t in rels:
        line = _link(rng, it, kind, s, t)
        link_lines.append(line)

    # ---- notes -------------------------------------------------------------------
    note_lines: list[str] = []
    if rng.random() < 0.25:
        texts = rng.sample(NOTE_TEXTS, 2)
        declared = [c for c in classes if c.style != "implicit"]  # PlantUML needs the class to exist
        for k, c in enumerate(rng.sample(declared, min(len(declared), rng.choice([1, 1, 2])))):
            text = texts[k]
            r = rng.random()
            pos = rng.choice(["left", "right", "top", "bottom"])
            if r < 0.6:
                note_lines.append(f"note {pos} of {c.key} : {text}")
            elif r < 0.85:  # block note, possibly over two lines
                parts = text.split(" ", 1)
                lines = parts if len(parts) == 2 and rng.random() < 0.6 else [text]
                text = "\n".join(lines)
                note_lines += [f"note {pos} of {c.key}"] + [f"  {x}" for x in lines] + ["end note"]
            else:
                alias = f"N{k + 1}"
                note_lines.append(f'note "{text}" as {alias}')
                note_lines.append(f"{alias} {rng.choice(['..', '.', '...'])} {c.key}")
            it.edge(it.node("note", text), c.nid, "anchor")

    # ---- code ----------------------------------------------------------------------
    decl_top = [ln for c in classes if c.pkg is None for ln in _declare(rng, c)]
    pkg_blocks = []
    for i, p in enumerate(packages):
        if p["parent"] is None:
            pkg_blocks += _package_block(rng, packages, i)
    member_lines = [f"{c.key} : {m}" for c in classes if c.style == "lines" for m in c.body
                    if m not in ("--", "..", "==", "__")]
    body = (pkg_blocks + decl_top) if rng.random() < 0.6 else (decl_top + pkg_blocks)
    body += member_lines
    if rng.random() < 0.5:
        body += link_lines + note_lines
    else:
        body += note_lines + link_lines
    return body, it.d


def _declare(rng: random.Random, c: _Cls, indent: str = "") -> list[str]:
    if c.style == "implicit":
        return []
    if c.kind == "abstract":
        kw = "abstract class" if rng.random() < 0.7 else "abstract"
    else:
        kw = c.kind
    head = f'{kw} "{c.display}" as {c.name}' if c.display else f"{kw} {c.name}{c.generic}"
    if c.stereos:
        head += " " + " ".join(f"<<{s}>>" for s in c.stereos)
    if c.style in ("bare", "lines") or not c.body:
        return [indent + head + (" {}" if c.style == "bare" and rng.random() < 0.3 else "")]
    pad = indent + rng.choice(["  ", "    ", "\t", ""])
    return [indent + head + " {"] + [pad + m for m in c.body] + [indent + "}"]


def _package_block(rng: random.Random, packages: list[dict], i: int, indent: str = "") -> list[str]:
    p = packages[i]
    kw = "namespace" if p["ns"] else "package"
    name = f'"{p["name"]}"' if p["quoted"] else p["name"]
    out = [f"{indent}{kw} {name} {{"]
    inner = indent + "  "
    for c in p["members"]:
        out += _declare(rng, c, inner)
    for j, q in enumerate(packages):
        if q["parent"] == i:
            out += _package_block(rng, packages, j, inner)
    out.append(f"{indent}}}")
    return out


def _relations(rng: random.Random, classes: list[_Cls]) -> list[tuple[str, _Cls, _Cls]]:
    """(IR kind, IR source, IR target): sub -> super, implementer -> interface, whole -> part."""
    rels: list[tuple[str, _Cls, _Cls]] = []
    pairs: set[frozenset] = set()

    def add(kind: str, s: _Cls, t: _Cls) -> bool:
        key = frozenset((s.name, t.name))
        if s is t or key in pairs:
            return False
        pairs.add(key)
        rels.append((kind, s, t))
        return True

    order = classes[:]
    rng.shuffle(order)
    rank = {c.name: k for k, c in enumerate(order)}
    concrete = [c for c in order if c.kind in ("class", "abstract")]
    ifaces = [c for c in order if c.kind == "interface"]
    enums = [c for c in order if c.kind == "enum"]
    for c in concrete:
        parents = [p for p in concrete if rank[p.name] < rank[c.name] and p is not c
                   and (p.kind == "abstract" or rng.random() < 0.5)]
        if parents and rng.random() < (0.45 if any(p.kind == "abstract" for p in parents) else 0.2):
            add("inherit", c, rng.choice(parents))
        if ifaces and rng.random() < 0.4:
            add("realize", c, rng.choice(ifaces))
    for c in ifaces:
        sup = [p for p in ifaces if rank[p.name] < rank[c.name]]
        if sup and rng.random() < 0.25:
            add("inherit", c, rng.choice(sup))
    for e in enums:
        if concrete:
            add(rng.choice(["assoc_dir", "assoc_dir", "depend", "assoc"]), rng.choice(concrete), e)
    linked = {x.name for _k, s, t in rels for x in (s, t)}
    for c in order:  # connect the rest
        if c.name in linked or len(classes) == 1:
            continue
        others = [o for o in order if o is not c and o.kind != "enum"] or [o for o in order if o is not c]
        o = rng.choice(others)
        if c.kind == "interface" or o.kind == "interface":
            impl, iface = (o, c) if c.kind == "interface" else (c, o)
            if impl.kind == "interface":
                add("inherit", c, o)
            elif rng.random() < 0.7:
                add("realize", impl, iface)
            else:
                add("depend", impl, iface)
        elif c.kind == "enum":
            add(rng.choice(["assoc_dir", "depend", "assoc"]), o, c)
        else:
            kind = rng.choices(["compose", "aggregate", "assoc", "assoc_dir", "depend"],
                               weights=(20, 15, 30, 20, 15))[0]
            s, t = (o, c) if rng.random() < 0.5 else (c, o)
            add(kind, s, t)
        linked |= {c.name, o.name}
    for _ in range(rng.randint(0, max(0, len(classes) // 3))):
        cands = [c for c in classes if c.kind in ("class", "abstract")]
        if len(cands) < 2:
            break
        s, t = rng.sample(cands, 2)
        add(rng.choices(["compose", "aggregate", "assoc", "assoc_dir", "depend"],
                        weights=(20, 15, 30, 20, 15))[0], s, t)
    rng.shuffle(rels)
    return rels


def _link(rng: random.Random, it: Intent, kind: str, s: _Cls, t: _Cls) -> str:
    """Write the relation in a random but equivalent spelling and add the intended edge."""
    dot = kind in ("realize", "depend")
    dash = "." if dot else "-"
    sl = tl = None
    if kind in ("compose", "aggregate", "assoc", "assoc_dir") and rng.random() < 0.5:
        sl = rng.choice(MULTS) if rng.random() < 0.8 else None
        tl = rng.choice(MULTS) if (sl is None or rng.random() < 0.8) else None
    label = ""
    if kind not in ("inherit", "realize") and rng.random() < 0.4:
        label = rng.choice(LINK_LABELS)
    elif kind in ("inherit", "realize") and rng.random() < 0.03:
        label = rng.choice(["extends", "implements"])
    r = rng.random()
    if label or sl or tl:  # Graphviz can drop labels of same-rank links: keep labelled ones vertical
        body = dash * 3 if r < 0.1 else dash * 2
    elif r < 0.12:
        body = f"{dash}{rng.choice(['up', 'down', 'left', 'right', 'u', 'd', 'l', 'r'])}{dash}"
    elif r < 0.22:
        body = dash
    elif r < 0.3:
        body = dash * 3
    else:
        body = dash * 2
    code_label = label
    if label and rng.random() < 0.3:
        code_label = f"{label} >" if rng.random() < 0.7 else f"< {label}"
    forward = rng.random() < 0.6
    head = {"inherit": "|>", "realize": "|>", "assoc_dir": ">", "depend": ">"}.get(kind, "")
    diamond = {"compose": "*", "aggregate": "o"}.get(kind, "")
    # left / right endpoints with their end labels and heads
    if kind in ("compose", "aggregate"):  # diamond at the whole (= IR source)
        if forward:
            parts = (s, sl, diamond, "", t, tl)
        else:
            parts = (t, tl, "", diamond, s, sl)
    elif head:
        if forward:
            parts = (s, sl, "", head, t, tl)
        else:
            parts = (t, tl, "<|" if head == "|>" else "<", "", s, sl)
    else:  # undirected association
        parts = (s, sl, "", "", t, tl) if forward else (t, tl, "", "", s, sl)
    left, lmult, lhead, rhead, right, rmult = parts
    out = [left.key]
    if lmult:
        out.append(f'"{lmult}"')
    out.append(f"{lhead}{body}{rhead}")
    if rmult:
        out.append(f'"{rmult}"')
    out.append(right.key)
    line = " ".join(out)
    if code_label:
        line += f" : {code_label}"
    it.edge(s.nid, t.nid, kind, label, sl, tl)
    return line
