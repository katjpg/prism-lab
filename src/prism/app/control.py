from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ControlKind = Literal[
    "float",
    "int",
    "bool",
    "select",
    "range",
]

ControlGroup = Literal[
    "basic",
    "advanced",
    "global",
]


@dataclass(frozen=True, slots=True)
class Option:
    label: str
    value: Any


@dataclass(frozen=True, slots=True)
class Control:
    name: str
    label: str
    kind: ControlKind
    default: Any

    group: ControlGroup = "basic"

    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None

    options: tuple[Option, ...] = ()
    help: str | None = None


@dataclass(frozen=True, slots=True)
class Mode:
    name: str
    label: str
    category: str

    controls: tuple[Control, ...]

    engine: str | None = None
    subtitle: str | None = None
    help: str | None = None


@dataclass(frozen=True, slots=True)
class Category:
    name: str
    label: str
    modes: tuple[Mode, ...]


def option(label: str, value: Any) -> Option:
    return Option(label=label, value=value)


def select_control(
    name: str,
    label: str,
    default: Any,
    options: tuple[Option, ...],
    *,
    group: ControlGroup = "basic",
    help: str | None = None,
) -> Control:
    return Control(
        name=name,
        label=label,
        kind="select",
        default=default,
        group=group,
        options=options,
        help=help,
    )


def float_control(
    name: str,
    label: str,
    default: float,
    min: float,
    max: float,
    *,
    step: float = 0.01,
    group: ControlGroup = "basic",
    help: str | None = None,
) -> Control:
    return Control(
        name=name,
        label=label,
        kind="float",
        default=default,
        min=min,
        max=max,
        step=step,
        group=group,
        help=help,
    )


def int_control(
    name: str,
    label: str,
    default: int,
    min: int,
    max: int,
    *,
    step: int = 1,
    group: ControlGroup = "basic",
    help: str | None = None,
) -> Control:
    return Control(
        name=name,
        label=label,
        kind="int",
        default=default,
        min=min,
        max=max,
        step=step,
        group=group,
        help=help,
    )


def bool_control(
    name: str,
    label: str,
    default: bool,
    *,
    group: ControlGroup = "basic",
    help: str | None = None,
) -> Control:
    return Control(
        name=name,
        label=label,
        kind="bool",
        default=default,
        group=group,
        help=help,
    )


def range_control(
    name: str,
    label: str,
    default: tuple[float, float],
    min: float,
    max: float,
    *,
    step: float = 0.01,
    group: ControlGroup = "advanced",
    help: str | None = None,
) -> Control:
    return Control(
        name=name,
        label=label,
        kind="range",
        default=default,
        min=min,
        max=max,
        step=step,
        group=group,
        help=help,
    )


def defaults_for(mode: Mode) -> dict[str, Any]:
    return {control.name: control.default for control in mode.controls}


def controls_by_group(
    mode: Mode,
    group: ControlGroup,
) -> tuple[Control, ...]:
    return tuple(control for control in mode.controls if control.group == group)


def find_control(
    mode: Mode,
    name: str,
) -> Control:
    for control in mode.controls:
        if control.name == name:
            return control

    raise KeyError(f"unknown control {name!r} for mode {mode.name!r}")


def resolve_params(
    mode: Mode,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = defaults_for(mode)

    if params:
        known = set(out)
        unknown = sorted(set(params) - known)

        if unknown:
            raise KeyError(f"unknown params for mode {mode.name!r}: {', '.join(unknown)}")

        out.update(params)

    return out


BACKGROUND = (
    option("White", "white"),
    option("Mean", "mean"),
    option("Source", "source"),
    option("Black", "black"),
    option("Canvas", "canvas"),
    option("Sketchbook", "sketchbook"),
    option("Black paper", "black-paper"),
)

STROKE_SHAPE = (
    option("Stroke", "stroke"),
    option("Dot", "dot"),
)

POSTER_FILL = (
    option("Value", "value"),
    option("Color", "color"),
)

POSTER_METHOD = (
    option("Multi-Otsu", "multiotsu"),
    option("K-Means", "kmeans"),
    option("Percentile", "percentile"),
)

SEGMENT_FILL = (
    option("Value", "value"),
    option("Color", "color"),
    option("Boundaries", "boundaries"),
)

SEGMENT_METHOD = (
    option("SLIC", "slic"),
    option("Watershed", "watershed"),
)

SHAPE_TYPE = (
    option("Ellipse", "ellipse"),
    option("Rectangle", "rectangle"),
    option("Triangle", "triangle"),
    option("Polygon", "polygon"),
    option("Combo", "combo"),
)

PALETTE_SORT = (
    option("Brightness", "lightness"),
    option("Tone", "tone"),
    option("Weight", "weight"),
)

RECOLOR_PALETTE = (
    option("Teal–Orange", "teal_orange"),
    option("Bleach", "bleach"),
    option("Sepia", "sepia"),
    option("Noir", "noir"),
    option("Warm Film", "warm_film"),
    option("Cool Film", "cool_film"),
)


def paint_common() -> tuple[Control, ...]:
    return (
        float_control(
            "color_variation",
            "Color Variation",
            0.10,
            0.0,
            0.30,
        ),
        float_control(
            "opacity",
            "Opacity",
            0.90,
            0.0,
            1.0,
        ),
        float_control(
            "saturation",
            "Saturation",
            1.10,
            0.50,
            2.00,
        ),
        select_control(
            "background",
            "Background",
            "white",
            BACKGROUND,
        ),
        float_control(
            "border",
            "Border",
            0.06,
            0.0,
            0.20,
        ),
    )


def paint_extras() -> tuple[Control, ...]:
    return (
        select_control(
            "stroke_shape",
            "Stroke Shape",
            "stroke",
            STROKE_SHAPE,
            group="advanced",
        ),
        float_control(
            "sensitivity",
            "Sensitivity",
            0.10,
            0.0,
            0.40,
            group="advanced",
        ),
        float_control(
            "spacing",
            "Spacing",
            1.00,
            0.50,
            2.00,
            group="advanced",
        ),
        float_control(
            "softness",
            "Softness",
            0.50,
            0.20,
            1.00,
            group="advanced",
        ),
        float_control(
            "min_stroke_length",
            "Min Stroke Length",
            4.0,
            0.0,
            40.0,
            group="advanced",
        ),
        float_control(
            "max_stroke_length",
            "Max Stroke Length",
            18.0,
            0.0,
            60.0,
            group="advanced",
        ),
    )


IMPRESSIONISM = Mode(
    name="impressionism",
    label="Impressionism",
    subtitle="Soft Brush",
    category="paint",
    engine="impressionist",
    controls=(
        float_control(
            "detail",
            "Detail",
            1.0,
            0.25,
            2.0,
        ),
        float_control(
            "stroke_size",
            "Stroke Size",
            1.0,
            0.25,
            2.0,
        ),
        float_control(
            "looseness",
            "Looseness",
            0.20,
            0.0,
            1.0,
        ),
        *paint_common(),
        *paint_extras(),
    ),
)

EXPRESSIONISM = Mode(
    name="expressionism",
    label="Expressionism",
    subtitle="Expressive Brush",
    category="paint",
    engine="expressionist",
    controls=(
        float_control(
            "intensity",
            "Intensity",
            1.0,
            0.25,
            2.0,
        ),
        float_control(
            "stroke_size",
            "Stroke Size",
            1.0,
            0.25,
            2.0,
        ),
        float_control(
            "gesture",
            "Gesture",
            0.75,
            0.0,
            1.0,
        ),
        *paint_common(),
        *paint_extras(),
    ),
)

COLORIST = Mode(
    name="colorist",
    label="Colorist",
    subtitle="Color Brush",
    category="paint",
    engine="colorist",
    controls=(
        float_control(
            "detail",
            "Detail",
            1.0,
            0.25,
            2.0,
        ),
        float_control(
            "stroke_size",
            "Stroke Size",
            1.0,
            0.25,
            2.0,
        ),
        float_control(
            "color_push",
            "Color Push",
            1.35,
            0.50,
            2.00,
        ),
        *paint_common(),
        *paint_extras(),
    ),
)

POINTILLISM = Mode(
    name="pointillism",
    label="Pointillism",
    subtitle="Dots",
    category="paint",
    engine="pointillist",
    controls=(
        float_control(
            "dot_size",
            "Dot Size",
            1.0,
            0.25,
            3.0,
        ),
        float_control(
            "contrast",
            "Contrast",
            1.15,
            0.5,
            2.0,
        ),
        float_control(
            "color_variation",
            "Color Variation",
            0.12,
            0.0,
            0.30,
        ),
        float_control(
            "opacity",
            "Opacity",
            0.92,
            0.0,
            1.0,
        ),
        select_control(
            "background",
            "Background",
            "white",
            BACKGROUND,
        ),
        float_control(
            "border",
            "Border",
            0.06,
            0.0,
            0.20,
        ),
        *paint_extras(),
    ),
)


def region_paint_controls() -> tuple[Control, ...]:
    return (
        float_control(
            "coverage",
            "Coverage",
            1.6,
            0.75,
            3.0,
        ),
        float_control(
            "edge_fit",
            "Edge Fit",
            2.2,
            2.0,
            14.0,
        ),
        float_control(
            "color_variation",
            "Color Variation",
            0.004,
            0.0,
            0.12,
        ),
        float_control(
            "saturation",
            "Saturation",
            1.06,
            0.50,
            2.00,
        ),
        select_control(
            "background",
            "Background",
            "white",
            BACKGROUND,
        ),
        float_control(
            "border",
            "Border",
            0.06,
            0.0,
            0.20,
        ),
        int_control(
            "palette_colors",
            "Palette Colors",
            14,
            2,
            24,
            group="advanced",
        ),
        float_control(
            "palette_mix",
            "Palette Mix",
            0.42,
            0.0,
            1.0,
            group="advanced",
        ),
        float_control(
            "color_smooth",
            "Color Smooth",
            1.9,
            0.0,
            4.0,
            group="advanced",
        ),
        float_control(
            "color_sat",
            "Color Saturation",
            1.12,
            0.50,
            2.00,
            group="advanced",
        ),
    )


GOUACHE = Mode(
    name="gouache",
    label="Gouache",
    subtitle="Opaque Brush",
    category="paint",
    engine="gouache",
    controls=region_paint_controls(),
)

ACRYLIC = Mode(
    name="acrylic",
    label="Acrylic",
    subtitle="Layered Brush",
    category="paint",
    engine="acrylic",
    controls=region_paint_controls(),
)

POSTERIZE = Mode(
    name="posterize",
    label="Posterize",
    category="abstract",
    engine="posterize",
    controls=(
        int_control(
            "levels",
            "Levels",
            5,
            2,
            8,
        ),
        select_control(
            "fill",
            "Fill",
            "value",
            POSTER_FILL,
        ),
        float_control(
            "smoothness",
            "Smoothness",
            0.25,
            0.0,
            1.0,
        ),
        select_control(
            "method",
            "Method",
            "multiotsu",
            POSTER_METHOD,
            group="advanced",
        ),
        bool_control(
            "perceived_value",
            "Perceived Value",
            True,
            group="advanced",
        ),
        bool_control(
            "even_lighting",
            "Even Lighting",
            False,
            group="advanced",
        ),
    ),
)

SEGMENT = Mode(
    name="segment",
    label="Segment",
    category="abstract",
    engine="segment",
    controls=(
        int_control(
            "regions",
            "Regions",
            600,
            50,
            4000,
        ),
        float_control(
            "edge_fit",
            "Edge Fit",
            10.0,
            1.0,
            40.0,
        ),
        select_control(
            "fill",
            "Fill",
            "value",
            SEGMENT_FILL,
        ),
        select_control(
            "method",
            "Method",
            "slic",
            SEGMENT_METHOD,
        ),
        float_control(
            "smoothness",
            "Smoothness",
            1.0,
            0.0,
            3.0,
        ),
        int_control(
            "markers",
            "Markers",
            250,
            20,
            1000,
            group="advanced",
        ),
        float_control(
            "watershed_tightness",
            "Watershed Tightness",
            0.001,
            0.0,
            0.01,
            group="advanced",
        ),
    ),
)

STIPPLE = Mode(
    name="stipple",
    label="Stipple",
    category="abstract",
    engine="stipple",
    controls=(
        int_control(
            "dots",
            "Dots",
            8000,
            500,
            50000,
        ),
        float_control(
            "dot_size",
            "Dot Size",
            0.75,
            0.50,
            3.00,
        ),
        float_control(
            "darkness",
            "Darkness",
            1.40,
            0.50,
            3.00,
        ),
        float_control(
            "edges",
            "Edges",
            0.35,
            0.0,
            1.0,
        ),
        float_control(
            "smoothness",
            "Smoothness",
            0.60,
            0.0,
            3.0,
            group="advanced",
        ),
        float_control(
            "paper_white",
            "Paper White",
            0.08,
            0.0,
            0.50,
            group="advanced",
        ),
    ),
)

GEOMETRIC = Mode(
    name="geometric",
    label="Geometric",
    category="abstract",
    engine="geometric",
    controls=(
        int_control(
            "shapes",
            "Shapes",
            120,
            10,
            1000,
        ),
        select_control(
            "shape_type",
            "Shape Type",
            "combo",
            SHAPE_TYPE,
        ),
        float_control(
            "opacity",
            "Opacity",
            0.60,
            0.0,
            1.0,
        ),
        float_control(
            "edges",
            "Edges",
            0.50,
            0.0,
            2.0,
        ),
        int_control(
            "search",
            "Search",
            8,
            1,
            32,
            group="advanced",
        ),
        float_control(
            "image_fit",
            "Image Fit",
            1.0,
            0.0,
            2.0,
            group="advanced",
        ),
        float_control(
            "effort",
            "Effort",
            0.45,
            0.0,
            1.0,
            group="advanced",
        ),
    ),
)

PALETTE = Mode(
    name="palette",
    label="Palette",
    category="color",
    engine="palette",
    controls=(
        int_control(
            "colors",
            "Colors",
            8,
            2,
            32,
        ),
        select_control(
            "sort",
            "Sort",
            "lightness",
            PALETTE_SORT,
        ),
        int_control(
            "precision",
            "Precision",
            20000,
            1000,
            100000,
        ),
    ),
)

RECOLOR = Mode(
    name="recolor",
    label="Recolor",
    category="color",
    engine="recolor",
    controls=(
        float_control(
            "strength",
            "Strength",
            0.40,
            0.0,
            1.0,
        ),
        select_control(
            "palette",
            "Palette",
            "teal_orange",
            RECOLOR_PALETTE,
        ),
        float_control(
            "vibrance",
            "Vibrance",
            0.85,
            0.0,
            2.0,
        ),
        float_control(
            "smoothness",
            "Smoothness",
            6.0,
            0.0,
            20.0,
        ),
    ),
)


PAINT = Category(
    name="paint",
    label="Paint",
    modes=(
        IMPRESSIONISM,
        EXPRESSIONISM,
        COLORIST,
        POINTILLISM,
        GOUACHE,
        ACRYLIC,
    ),
)

ABSTRACT = Category(
    name="abstract",
    label="Abstract",
    modes=(
        POSTERIZE,
        SEGMENT,
        STIPPLE,
        GEOMETRIC,
    ),
)

COLOR = Category(
    name="color",
    label="Color",
    modes=(
        PALETTE,
        RECOLOR,
    ),
)

CATEGORIES = (
    PAINT,
    ABSTRACT,
    COLOR,
)

CategoryName = Literal["paint", "abstract", "color"]

MODES = {mode.name: mode for category in CATEGORIES for mode in category.modes}

MODES_BY_CATEGORY = {
    category.name: {mode.name: mode for mode in category.modes} for category in CATEGORIES
}


def get_category(name: CategoryName) -> Category:
    for category in CATEGORIES:
        if category.name == name:
            return category

    raise KeyError(f"unknown category: {name!r}")


def get_mode(
    category: CategoryName,
    mode: str,
) -> Mode:
    try:
        return MODES_BY_CATEGORY[category][mode]
    except KeyError as exc:
        raise KeyError(f"unknown mode {category!r}/{mode!r}") from exc


__all__ = [
    "ABSTRACT",
    "ACRYLIC",
    "BACKGROUND",
    "CATEGORIES",
    "COLOR",
    "COLORIST",
    "Category",
    "CategoryName",
    "Control",
    "ControlGroup",
    "ControlKind",
    "EXPRESSIONISM",
    "GEOMETRIC",
    "GOUACHE",
    "IMPRESSIONISM",
    "MODES",
    "MODES_BY_CATEGORY",
    "Mode",
    "Option",
    "PAINT",
    "PALETTE",
    "POINTILLISM",
    "POSTERIZE",
    "RECOLOR",
    "SEGMENT",
    "STIPPLE",
    "bool_control",
    "controls_by_group",
    "defaults_for",
    "find_control",
    "float_control",
    "get_category",
    "get_mode",
    "int_control",
    "option",
    "range_control",
    "resolve_params",
    "select_control",
]
