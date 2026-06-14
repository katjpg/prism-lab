from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

Detail = Literal["draft", "standard", "high", "ultra"]


@dataclass(frozen=True)
class DetailPreset:
    side: int
    samples: int
    bristles: int
    grid: float
    radii: float


PRESETS: dict[Detail, DetailPreset] = {
    "draft": DetailPreset(
        side=640,
        samples=8,
        bristles=32,
        grid=1.05,
        radii=1.00,
    ),
    "standard": DetailPreset(
        side=900,
        samples=10,
        bristles=44,
        grid=0.88,
        radii=0.86,
    ),
    "high": DetailPreset(
        side=1280,
        samples=12,
        bristles=56,
        grid=0.76,
        radii=1.20,
    ),
    "ultra": DetailPreset(
        side=1600,
        samples=14,
        bristles=70,
        grid=0.66,
        radii=1.50,
    ),
}


def preset(detail: Detail) -> DetailPreset:
    try:
        return PRESETS[detail]
    except KeyError as exc:
        raise ValueError(f"unknown detail: {detail!r}") from exc


def fit_side(
    image: np.ndarray,
    side: int,
) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]

    if side <= 0 or max(h, w) <= side:
        return image.astype("float32", copy=False), 1.0

    down = side / max(h, w)
    hh = max(1, round(h * down))
    ww = max(1, round(w * down))

    out = cv2.resize(
        image.astype("float32", copy=False),
        (ww, hh),
        interpolation=cv2.INTER_AREA,
    )

    up = max(h, w) / max(hh, ww)
    return out.astype("float32", copy=False), float(up)


def fit_pixels(
    image: np.ndarray,
    pixels: tuple[int, int] | None,
) -> tuple[np.ndarray, float]:
    if pixels is None:
        return image.astype("float32", copy=False), 1.0

    w, h = pixels
    if w <= 0 or h <= 0:
        raise ValueError("pixels must be positive")

    h0, w0 = image.shape[:2]

    if (w0, h0) == (w, h):
        return image.astype("float32", copy=False), 1.0

    out = cv2.resize(
        image.astype("float32", copy=False),
        (w, h),
        interpolation=cv2.INTER_AREA if w * h < w0 * h0 else cv2.INTER_LINEAR,
    )

    up = max(w0 / w, h0 / h)
    return out.astype("float32", copy=False), float(up)


def fit_image(
    image: np.ndarray,
    detail: Detail,
    pixels: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    if pixels is not None:
        image, _ = fit_pixels(image, pixels)
    return fit_side(image, preset(detail).side)


def scaled_radii(
    radii: tuple[int, ...],
    detail: Detail,
) -> tuple[int, ...]:
    p = preset(detail)
    out = tuple(max(1, round(r * p.radii)) for r in radii)
    return tuple(dict.fromkeys(out))


__all__ = [
    "PRESETS",
    "Detail",
    "DetailPreset",
    "fit_image",
    "fit_pixels",
    "fit_side",
    "preset",
    "scaled_radii",
]
