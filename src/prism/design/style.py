from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PainterlyStyle = Literal[
    "impressionist",
    "expressionist",
    "colorist",
    "pointillist",
]

StrokeShape = Literal["stroke", "dot"]
Background = Literal["mean", "white", "black", "source"]


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
            background="mean",
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
            background="mean",
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
            background="mean",
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


__all__ = [
    "Background",
    "PaintConfig",
    "PainterlyStyle",
    "StrokeShape",
    "style_config",
]
