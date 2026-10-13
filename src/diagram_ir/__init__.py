"""Diagram IR - one structured representation for flowcharts, UML, ER, state machines,
logic circuits and generic graphs, with converters to DGM (model output), JSON, YAML, XML.

    from diagram_ir import load, dump, canonicalize, validate
    d = load(model_output)                 # DGM / JSON / YAML / XML, auto-detected
    print(dump(canonicalize(d), "json"))

See SPEC.md for the format definition.
"""

from .canonical import canonicalize, normalize_text
from .convert import (
    FORMATS,
    convert,
    detect_format,
    dump,
    from_dict,
    from_json,
    from_xml,
    from_yaml,
    load,
    to_dict,
    to_json,
    to_xml,
    to_yaml,
)
from .dsl import DgmSyntaxError, dumps, loads, parse
from .model import (
    DIAGRAM_TYPES,
    FORMAT_NAME,
    VERSION,
    Diagram,
    Edge,
    Group,
    Issue,
    Member,
    Node,
    errors,
    validate,
)

__all__ = [
    "DIAGRAM_TYPES", "FORMATS", "FORMAT_NAME", "VERSION",
    "Diagram", "Edge", "Group", "Member", "Node", "Issue", "DgmSyntaxError",
    "canonicalize", "convert", "detect_format", "dump", "dumps", "errors", "from_dict",
    "from_json", "from_xml", "from_yaml", "load", "loads", "normalize_text", "parse",
    "to_dict", "to_json", "to_xml", "to_yaml", "validate",
]
