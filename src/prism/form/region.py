from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries, slic, watershed

from prism.color.value import value_channel


RegionMode = Literal["boundaries", "value", "color"]
RegionMethod = Literal["slic", "watershed"]

DEFAULT_N_SEGMENTS = 600
DEFAULT_COMPACTNESS = 10.0
DEFAULT_SMOOTH = 1.0

DEFAULT_WATERSHED_MARKERS = 250
DEFAULT_WATERSHED_COMPACTNESS = 0.001


@dataclass
class RegionMap:
    image: np.ndarray
    labels: np.ndarray


def superpixels(
    rgb: np.ndarray,
    n_segments: int = DEFAULT_N_SEGMENTS,
    compactness: float = DEFAULT_COMPACTNESS,
    smooth: float = DEFAULT_SMOOTH,
) -> np.ndarray:
    return slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=smooth,
        slic_zero=True,
        start_label=1,
    ).astype("int64")


def watershed_regions(
    rgb: np.ndarray,
    markers: int = DEFAULT_WATERSHED_MARKERS,
    smooth: float = DEFAULT_SMOOTH,
    compactness: float = DEFAULT_WATERSHED_COMPACTNESS,
) -> np.ndarray:
    value = value_channel(rgb)

    if smooth > 0:
        value = cv2.GaussianBlur(value.astype("float32"), ksize=(0, 0), sigmaX=smooth)

    gradient = sobel(value).astype("float32")

    labels = watershed(gradient, markers=markers, compactness=compactness)  # type: ignore
    return labels.astype("int64")


def region_labels(
    rgb: np.ndarray,
    method: RegionMethod = "slic",
    n_segments: int = DEFAULT_N_SEGMENTS,
    compactness: float = DEFAULT_COMPACTNESS,
    smooth: float = DEFAULT_SMOOTH,
    markers: int = DEFAULT_WATERSHED_MARKERS,
    watershed_compactness: float = DEFAULT_WATERSHED_COMPACTNESS,
) -> np.ndarray:
    if method == "slic":
        return superpixels(
            rgb,
            n_segments=n_segments,
            compactness=compactness,
            smooth=smooth,
        )

    if method == "watershed":
        return watershed_regions(
            rgb,
            markers=markers,
            smooth=smooth,
            compactness=watershed_compactness,
        )

    raise ValueError(f"unknown region method: {method!r}")


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
    method: RegionMethod = "slic",
    n_segments: int = DEFAULT_N_SEGMENTS,
    compactness: float = DEFAULT_COMPACTNESS,
    smooth: float = DEFAULT_SMOOTH,
    markers: int = DEFAULT_WATERSHED_MARKERS,
    watershed_compactness: float = DEFAULT_WATERSHED_COMPACTNESS,
) -> RegionMap:
    labels = region_labels(
        rgb,
        method=method,
        n_segments=n_segments,
        compactness=compactness,
        smooth=smooth,
        markers=markers,
        watershed_compactness=watershed_compactness,
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


__all__ = [
    "RegionMap",
    "RegionMethod",
    "RegionMode",
    "mean_color_by_region",
    "mean_value_by_region",
    "region_labels",
    "region_map",
    "superpixels",
    "watershed_regions",
]
