from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal, Protocol

import numpy as np

from prism.paint.ground import Background
from prism.preset import Detail

PainterlyStyle = Literal[
    "impressionist",
    "expressionist",
    "colorist",
    "pointillist",
]

StrokeShape = Literal["stroke", "dot"]


@dataclass(frozen=True)
class PaintConfig:
    radii: tuple[int, ...]
    threshold: float
    alpha: float
    grid_factor: float
    blur_factor: float
    min_length: float
    max_length: float
    color_jitter: float
    angle_jitter: float
    saturation: float
    stroke_shape: StrokeShape
    background: Background


@dataclass(frozen=True, slots=True)
class RegionPaintConfig:
    coverage: float | None = None
    compactness: float = 5.0
    alpha: float = 1.0
    saturation: float = 1.06
    background: str = "white"
    color_jitter: float = 0.004
    palette_colors: int = 16
    palette_mix: float = 0.42
    color_smooth: float = 1.5
    palette_saturation: float = 1.15


@dataclass
class PaintResult:
    image: np.ndarray
    style: str
    config: PaintConfig | RegionPaintConfig
    detail: Detail = "standard"
    alpha: np.ndarray | None = None


def style_config(style: PainterlyStyle) -> PaintConfig:
    if style == "impressionist":
        return PaintConfig(
            radii=(12, 8, 4, 2),
            threshold=0.12,
            alpha=0.88,
            grid_factor=1.0,
            blur_factor=0.5,
            min_length=4.0,
            max_length=18.0,
            color_jitter=0.04,
            angle_jitter=0.20,
            saturation=1.05,
            stroke_shape="stroke",
            background="white",
        )

    if style == "expressionist":
        return PaintConfig(
            radii=(18, 12, 7, 3),
            threshold=0.10,
            alpha=0.92,
            grid_factor=1.0,
            blur_factor=0.45,
            min_length=6.0,
            max_length=28.0,
            color_jitter=0.16,
            angle_jitter=0.75,
            saturation=1.25,
            stroke_shape="stroke",
            background="white",
        )

    if style == "colorist":
        return PaintConfig(
            radii=(16, 10, 5, 2),
            threshold=0.11,
            alpha=0.86,
            grid_factor=1.0,
            blur_factor=0.55,
            min_length=4.0,
            max_length=20.0,
            color_jitter=0.18,
            angle_jitter=0.35,
            saturation=1.35,
            stroke_shape="stroke",
            background="white",
        )

    if style == "pointillist":
        return PaintConfig(
            radii=(4, 3, 2),
            threshold=0.07,
            alpha=0.92,
            grid_factor=0.75,
            blur_factor=0.35,
            min_length=0.0,
            max_length=0.0,
            color_jitter=0.12,
            angle_jitter=0.0,
            saturation=1.15,
            stroke_shape="dot",
            background="white",
        )

    raise ValueError(f"unknown painterly style: {style!r}")


def build_paint_config(
    engine_style: str,
    mode: str,
    params: dict[str, Any],
) -> PaintConfig:
    cfg = style_config(engine_style)  # type: ignore[arg-type]
    base: dict[str, Any] = asdict(cfg)

    radii_scale = float(
        params.get("stroke_size") or params.get("detail") or params.get("intensity") or 1.0
    )
    base["radii"] = tuple(max(1, round(r * radii_scale)) for r in cfg.radii)

    base["alpha"] = float(params.get("opacity", cfg.alpha))
    base["color_jitter"] = float(params.get("color_variation", cfg.color_jitter))
    base["saturation"] = float(params.get("saturation", cfg.saturation))
    base["background"] = str(params.get("background", cfg.background))
    base["threshold"] = float(params.get("sensitivity", cfg.threshold))
    base["grid_factor"] = float(params.get("spacing", cfg.grid_factor))
    base["blur_factor"] = float(params.get("softness", cfg.blur_factor))
    base["min_length"] = float(params.get("min_stroke_length", cfg.min_length))
    base["max_length"] = float(params.get("max_stroke_length", cfg.max_length))
    base["angle_jitter"] = float(params.get("looseness", cfg.angle_jitter))

    if mode == "expressionism":
        base["angle_jitter"] = float(params["gesture"])
    elif mode == "colorist":
        base["saturation"] = float(params["color_push"])
    elif mode == "pointillism":
        base["stroke_shape"] = "dot"
        base["saturation"] = float(params["contrast"])
        base["radii"] = tuple(max(1, round(r * float(params["dot_size"]))) for r in cfg.radii)

    if "stroke_shape" in params and mode != "pointillism":
        base["stroke_shape"] = str(params["stroke_shape"])

    return PaintConfig(**base)


class BrushStyle(Protocol):
    category: ClassVar[str]
    mode: ClassVar[str]

    def params(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Acrylic:
    category: ClassVar[str] = "paint"
    mode: ClassVar[str] = "acrylic"

    coverage: float = 1.6
    edge_fit: float = 2.2
    color_variation: float = 0.004
    saturation: float = 1.06
    background: Background = "white"
    border: float = 0.06
    palette_colors: int = 14
    palette_mix: float = 0.42
    color_smooth: float = 1.9
    color_sat: float = 1.12

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Gouache:
    category: ClassVar[str] = "paint"
    mode: ClassVar[str] = "gouache"

    coverage: float = 1.6
    edge_fit: float = 2.2
    color_variation: float = 0.004
    saturation: float = 1.06
    background: Background = "white"
    border: float = 0.06
    palette_colors: int = 14
    palette_mix: float = 0.42
    color_smooth: float = 1.9
    color_sat: float = 1.12

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Impressionist:
    category: ClassVar[str] = "paint"
    mode: ClassVar[str] = "impressionism"

    detail: float = 1.0
    stroke_size: float = 1.0
    looseness: float = 0.2
    color_variation: float = 0.1
    opacity: float = 0.9
    saturation: float = 1.1
    background: Background = "white"
    border: float = 0.06
    stroke_shape: StrokeShape = "stroke"
    sensitivity: float = 0.1
    spacing: float = 1.0
    softness: float = 0.5
    min_stroke_length: float = 4.0
    max_stroke_length: float = 18.0

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Expressionist:
    category: ClassVar[str] = "paint"
    mode: ClassVar[str] = "expressionism"

    intensity: float = 1.0
    stroke_size: float = 1.0
    gesture: float = 0.75
    color_variation: float = 0.1
    opacity: float = 0.9
    saturation: float = 1.1
    background: Background = "white"
    border: float = 0.06
    stroke_shape: StrokeShape = "stroke"
    sensitivity: float = 0.1
    spacing: float = 1.0
    softness: float = 0.5
    min_stroke_length: float = 4.0
    max_stroke_length: float = 18.0

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Colorist:
    category: ClassVar[str] = "paint"
    mode: ClassVar[str] = "colorist"

    detail: float = 1.0
    stroke_size: float = 1.0
    color_push: float = 1.35
    color_variation: float = 0.1
    opacity: float = 0.9
    saturation: float = 1.1
    background: Background = "white"
    border: float = 0.06
    stroke_shape: StrokeShape = "stroke"
    sensitivity: float = 0.1
    spacing: float = 1.0
    softness: float = 0.5
    min_stroke_length: float = 4.0
    max_stroke_length: float = 18.0

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Pointillist:
    category: ClassVar[str] = "paint"
    mode: ClassVar[str] = "pointillism"

    dot_size: float = 1.0
    contrast: float = 1.15
    color_variation: float = 0.12
    opacity: float = 0.92
    background: Background = "white"
    border: float = 0.06
    stroke_shape: StrokeShape = "stroke"
    sensitivity: float = 0.1
    spacing: float = 1.0
    softness: float = 0.5
    min_stroke_length: float = 4.0
    max_stroke_length: float = 18.0

    def params(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "Acrylic",
    "Background",
    "BrushStyle",
    "Colorist",
    "Expressionist",
    "Gouache",
    "Impressionist",
    "PaintConfig",
    "PaintResult",
    "PainterlyStyle",
    "Pointillist",
    "RegionPaintConfig",
    "StrokeShape",
    "build_paint_config",
    "style_config",
]
