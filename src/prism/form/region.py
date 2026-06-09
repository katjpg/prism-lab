from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from skimage.segmentation import mark_boundaries, slic

from prism.color.value import value_channel


RegionMode = Literal["boundaries", "value", "color"]


@dataclass
class RegionMap:
    image: np.ndarray
    labels: np.ndarray


def superpixels(
    rgb: np.ndarray,
    n_segments: int = 600,
    compactness: float = 10.0,
    smooth: float = 1.0,
) -> np.ndarray:
    return slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=smooth,
        slic_zero=True,
        start_label=1,
    )


def mean_value_by_region(
    labels: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    flat = labels.ravel()
    n = int(flat.max()) + 1

    sums = np.bincount(
        flat,
        weights=value.ravel().astype("float64"),
        minlength=n,
    )
    counts = np.maximum(np.bincount(flat, minlength=n), 1)

    return np.asarray((sums / counts)[labels], dtype="float32")


def mean_color_by_region(
    labels: np.ndarray,
    rgb: np.ndarray,
) -> np.ndarray:
    flat = labels.ravel()
    n = int(flat.max()) + 1

    counts = np.maximum(np.bincount(flat, minlength=n), 1)[:, None]

    means = (
        np.stack(
            [
                np.bincount(
                    flat,
                    weights=rgb[..., c].ravel().astype("float64"),
                    minlength=n,
                )
                for c in range(3)
            ],
            axis=1,
        )
        / counts
    )

    return np.asarray(means[labels], dtype="float32")


def region_map(
    rgb: np.ndarray,
    mode: RegionMode = "value",
    n_segments: int = 600,
    compactness: float = 10.0,
    smooth: float = 1.0,
) -> RegionMap:
    labels = superpixels(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        smooth=smooth,
    )

    if mode == "boundaries":
        image = np.asarray(
            mark_boundaries(rgb, labels, color=(1, 0, 0)),
            dtype="float32",
        )
    elif mode == "value":
        image = mean_value_by_region(labels, value_channel(rgb))
    elif mode == "color":
        image = mean_color_by_region(labels, rgb)
    else:
        raise ValueError(f"unknown region mode: {mode!r}")

    return RegionMap(image=image, labels=labels)
