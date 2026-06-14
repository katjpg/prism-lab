from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from prism.color.value import DEFAULT_SEED, Value, ValueMethod
from prism.preset import Detail, fit_image
from prism.raster.region import mean_color_by_label

ValueFill = Literal["value", "color"]


@dataclass
class ValueMap:
    image: np.ndarray
    labels: np.ndarray
    n_bands: int


def value_map(
    rgb: np.ndarray,
    *,
    n_bands: int,
    method: ValueMethod,
    chroma: bool,
    flatten: bool,
    smooth: float,
    fill: ValueFill,
    seed: int,
) -> ValueMap:
    result = Value(
        n_bands=n_bands,
        method=method,
        chroma=chroma,
        smooth=smooth > 0,
        flatten=flatten,
        seed=seed,
    ).extract(rgb)

    if fill == "value":
        image = result.scaffold
    elif fill == "color":
        image = mean_color_by_label(result.labels, rgb)
    else:
        raise ValueError(f"unknown value fill: {fill!r}")

    return ValueMap(
        image=image,
        labels=result.labels,
        n_bands=result.n_bands,
    )


def quantize_value(
    rgb: np.ndarray,
    n_bands: int = 5,
    smooth: float = 0.25,
    seed: int = DEFAULT_SEED,
    detail: Detail = "standard",
    pixels: tuple[int, int] | None = None,
) -> ValueMap:
    rgb, _ = fit_image(rgb, detail, pixels)

    return value_map(
        rgb,
        n_bands=n_bands,
        method="multiotsu",
        chroma=True,
        flatten=False,
        smooth=smooth,
        fill="value",
        seed=seed,
    )
