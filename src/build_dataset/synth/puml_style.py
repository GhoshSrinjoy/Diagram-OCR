"""Random visual style for the synthetic PlantUML diagrams: built-in theme, sketch mode, fonts,
colours, line routing, layout direction and per-kind skinparams.

Every option below was rendered with PlantUML 1.2026.8 and checked to draw no text of its own
(no deprecation / warning banner) and to keep all diagram text readable. Rejected on purpose:
  * `skinparam handwritten true`, `skinparam padding`, `skinparam ParticipantPadding`: PlantUML
    1.2026 draws a "please use ..." banner into the image (`!option handwritten true` is clean);
  * `skinparam componentStyle rectangle`: components become plain boxes (kind not visible);
  * `skinparam maxMessageSize` / `wrapWidth`: wrap text at places the target does not show;
  * `skinparam conditionStyle diamond`: drops the labels of `repeat while (c) is (x) not (y)`;
  * `left to right direction` in sequence / activity diagrams: a syntax error there;
  * themes outside THEMES[kind] (see there).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

# Most bootswatch-style themes leave the page transparent (they expect the viewer's page colour). A
# transparent PNG turns black under PIL's convert("RGB") - what the training loader does - so these
# get an explicit page colour: white for the light ones, the theme's intended dark for the others.
PAGE = {
    "aws-orange": "white", "carbon-gray": "white", "cerulean": "white", "cerulean-outline": "white",
    "cloudscape-design": "white", "materia": "white", "materia-outline": "white", "sketchy-outline": "white",
    "cyborg": "#060606", "cyborg-outline": "#060606", "hacker": "#222222", "superhero": "#2B3E50",
    "superhero-outline": "#2B3E50",
}
_OPAQUE = ("amiga", "blueprint", "crt-amber", "crt-green", "mars", "mimeograph", "mono", "plain",
           "reddress-darkblue", "reddress-darkgreen", "reddress-darkorange", "reddress-darkred",
           "reddress-lightblue", "reddress-lightgreen", "reddress-lightorange", "reddress-lightred",
           "sunlust", "toy", "vibrant")
_LIGHT_PAGED = ("aws-orange", "carbon-gray", "cerulean", "cerulean-outline")
# Themes per kind whose renders keep every text readable (fill / local background contrast >= 1.6 for
# every text of 16 generated diagrams per kind and theme, 10th percentile >= 2.3), keep connectors
# visible (stroke / page contrast), draw nothing that is not in the target, and passed spot checks by
# eye. Rejected everywhere: spacelab-white, minty, sandstone, sketchy, spacelab, united, metal (white or
# light text on white page parts: actor / lifeline names, fragment and boundary titles, notes),
# lightgray, silver, black-knight (connectors almost invisible on their page), bluegray (faint text).
# The rest is kind-specific (e.g. materia use case boundary titles, cyborg / superhero fragment headers).
_EXTRA = {
    "class": ("cloudscape-design", "materia-outline", "sketchy-outline", "cyborg-outline"),
    "sequence": ("cloudscape-design", "materia", "materia-outline", "sketchy-outline"),
    "activity": ("cloudscape-design", "materia", "materia-outline", "sketchy-outline", "cyborg",
                 "cyborg-outline", "hacker", "superhero", "superhero-outline"),
    "state": ("cloudscape-design", "materia", "cyborg", "cyborg-outline", "superhero", "superhero-outline"),
    "usecase": ("cloudscape-design", "materia-outline", "sketchy-outline", "cyborg", "cyborg-outline",
                "hacker", "superhero", "superhero-outline"),
    "component": ("materia-outline", "cyborg-outline", "hacker", "superhero-outline"),
    "deployment": ("materia", "materia-outline", "cyborg-outline", "hacker", "superhero-outline"),
}
THEMES: dict[str, tuple[str, ...]] = {k: _OPAQUE + _LIGHT_PAGED + v for k, v in _EXTRA.items()}
DARK = frozenset({"amiga", "blueprint", "crt-amber", "crt-green", "cyborg", "cyborg-outline",
                  "hacker", "superhero", "superhero-outline"})

HAND_FONTS = ("Ink Free", "Segoe Print", "Comic Sans MS", "Segoe Script", "Bradley Hand ITC", "MV Boli")
PLAIN_FONTS = ("Arial", "Verdana", "Tahoma", "Trebuchet MS", "Georgia", "Times New Roman", "Courier New",
               "Consolas", "Calibri", "Cambria", "Candara", "Segoe UI", "Lucida Console")
LIGHT_BACKGROUNDS = ("#FFFFF0", "#F8F8FF", "#FDF6E3", "#F0FFF0", "#FFF8F0", "#F5F5F5", "#EEF5FF", "#FFFDE7")
DARK_INKS = ("#333333", "#1F4E79", "#7B241C", "#145A32", "#4A235A", "#000080", "#2F4F4F", "#5D4037")
LIGHT_FILLS = ("#FEFECE", "#E3F2FD", "#E8F5E9", "#FFF3E0", "#F3E5F5", "#FFFFFF", "#ECEFF1", "#FFF9C4")
GRAPH_KINDS = ("class", "state", "usecase", "component", "deployment")
LR_RATE = {"class": 0.18, "state": 0.22, "usecase": 0.5, "component": 0.35, "deployment": 0.35}


@dataclass
class Style:
    lines: list[str] = field(default_factory=list)
    theme: str = "default"
    handwritten: bool = False
    direction: Optional[str] = None  # "LR" when `left to right direction` is set


def make_style(rng: random.Random, kind: str, has_lanes: bool = False) -> Style:
    """`kind`: class / sequence / activity / state / usecase / component / deployment."""
    st = Style()
    out = st.lines
    if rng.random() < 0.55:
        dark = [t for t in THEMES[kind] if t in DARK]
        light = [t for t in THEMES[kind] if t not in DARK]
        st.theme = rng.choice(dark if rng.random() < 0.2 else light)
        out.append(f"!theme {st.theme}")
        if st.theme in PAGE:
            out.append(f"skinparam backgroundColor {PAGE[st.theme]}")
    if rng.random() < 0.15:
        st.handwritten = True
        out.append("!option handwritten true")
    if kind == "sequence" and rng.random() < 0.12:
        out.append("!pragma teoz true")
    if kind == "activity" and rng.random() < 0.1:
        out.append("!pragma useVerticalIf on")
    r = rng.random()
    if r < 0.07:
        out.append("skinparam monochrome true")
    elif r < 0.085 and st.theme == "default":  # black page: not on top of a theme's page colour
        out.append("skinparam monochrome reverse")
    if rng.random() < 0.35:
        font = rng.choice(HAND_FONTS) if rng.random() < 0.45 else rng.choice(PLAIN_FONTS)
        out.append(f'skinparam defaultFontName "{font}"')
    if rng.random() < 0.25:
        out.append(f"skinparam defaultFontSize {rng.randint(10, 17)}")
    if rng.random() < 0.15:
        out.append(f"skinparam shadowing {rng.choice(['true', 'false'])}")
    if rng.random() < 0.12:
        out.append(f"skinparam roundCorner {rng.randint(4, 25)}")
    if st.theme == "default":  # colours only on the default look: themes bring their own palette
        if rng.random() < 0.1:
            out.append(f"skinparam backgroundColor {rng.choice(LIGHT_BACKGROUNDS)}")
        if rng.random() < 0.1:
            out.append(f"skinparam ArrowColor {rng.choice(DARK_INKS)}")
    if kind in GRAPH_KINDS:
        r = rng.random()
        if r < 0.12:
            out.append("skinparam linetype ortho")
        elif r < 0.2:
            out.append("skinparam linetype polyline")
        if rng.random() < LR_RATE[kind]:
            out.append("left to right direction")
            st.direction = "LR"
    if rng.random() < 0.08:
        out.append(f"scale {rng.choice(['0.9', '1.2', '1.3', '1.5'])}")
    _kind_params(rng, kind, st, has_lanes)
    return st


def _kind_params(rng: random.Random, kind: str, st: Style, has_lanes: bool) -> None:
    out = st.lines
    plain = st.theme == "default"
    if kind == "class":
        if rng.random() < 0.35:
            out.append("skinparam classAttributeIconSize 0")
        if rng.random() < 0.25:
            out.append(rng.choice(["hide empty members", "hide empty methods", "hide empty fields"]))
        if rng.random() < 0.1:
            out.append(f"skinparam packageStyle {rng.choice(['rectangle', 'frame', 'folder', 'node'])}")
        if plain and rng.random() < 0.12:
            out.append(f"skinparam class {{\nBackgroundColor {rng.choice(LIGHT_FILLS)}\n"
                       f"BorderColor {rng.choice(DARK_INKS)}\n}}")
    elif kind == "sequence":
        if rng.random() < 0.35:
            out.append("hide footbox")
        if rng.random() < 0.1:
            out.append(f"skinparam sequenceMessageAlign {rng.choice(['center', 'right', 'direction'])}")
        if rng.random() < 0.1:
            out.append("skinparam responseMessageBelowArrow true")
        if rng.random() < 0.08:
            out.append("skinparam style strictuml")
        if rng.random() < 0.05:
            out.append("skinparam lifelineStrategy solid")
        if plain and rng.random() < 0.1:
            out.append(f"skinparam sequence {{\nArrowColor {rng.choice(DARK_INKS)}\n"
                       f"LifeLineBorderColor {rng.choice(DARK_INKS)}\n"
                       f"ParticipantBackgroundColor {rng.choice(LIGHT_FILLS)}\n}}")
    elif kind == "activity":
        r = rng.random()
        if r < 0.12:  # (not `diamond`: it drops the `repeat while ... is (x) not (y)` labels)
            out.append("skinparam conditionStyle InsideDiamond")
        elif r < 0.2:
            out.append("skinparam conditionStyle foo1")
        if has_lanes and rng.random() < 0.25:
            out.append("skinparam swimlaneWidth same")
        if plain and rng.random() < 0.12:
            out.append(f"skinparam activity {{\nBackgroundColor {rng.choice(LIGHT_FILLS)}\n"
                       f"BorderColor {rng.choice(DARK_INKS)}\n}}")
    elif kind == "state":
        if rng.random() < 0.35:
            out.append("hide empty description")
        if plain and rng.random() < 0.12:
            out.append(f"skinparam state {{\nBackgroundColor {rng.choice(LIGHT_FILLS)}\n"
                       f"BorderColor {rng.choice(DARK_INKS)}\n}}")
    elif kind == "usecase":
        r = rng.random()
        if r < 0.2:
            out.append("skinparam actorStyle awesome")
        elif r < 0.3:
            out.append("skinparam actorStyle hollow")
        if rng.random() < 0.15:
            out.append("skinparam packageStyle rectangle")
        if plain and rng.random() < 0.12:
            out.append(f"skinparam usecase {{\nBackgroundColor {rng.choice(LIGHT_FILLS)}\n"
                       f"BorderColor {rng.choice(DARK_INKS)}\n}}")
    elif kind in ("component", "deployment"):
        if rng.random() < 0.2:
            out.append("skinparam componentStyle uml1")
        if plain and rng.random() < 0.12:
            out.append(f"skinparam node {{\nBackgroundColor {rng.choice(LIGHT_FILLS)}\n}}")
