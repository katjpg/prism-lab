from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from prism.color.value import EPS, DEFAULT_SEED, Value


@dataclass
class ValueMap:
    image: np.ndarray
    labels: np.ndarray
    color: np.ndarray
    n_bands: int


def preprocess_rgb(
    rgb: np.ndarray,
    denoise: float = 0.15,
    smooth: float = 0.25,
) -> np.ndarray:
    if denoise <= 0 and smooth <= 0:
        return rgb.astype("float32")

    diag = float(np.hypot(*rgb.shape[:2]))

    out = cv2.bilateralFilter(
        rgb.astype("float32"),
        d=0,
        sigmaColor=float(max(denoise, EPS)),
        sigmaSpace=max(1.0, 0.01 * diag),
    )

    if smooth > 0:
        blur = cv2.GaussianBlur(
            out,
            ksize=(0, 0),
            sigmaX=max(0.5, 2.0 * smooth),
        )
        out = (1.0 - smooth) * out + smooth * blur

    return np.clip(out, 0.0, 1.0).astype("float32")


def mean_color_by_value(
    rgb: np.ndarray,
    labels: np.ndarray,
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


def quantize_value(
    rgb: np.ndarray,
    n_bands: int = 5,
    smooth: float = 0.25,
    seed: int = DEFAULT_SEED,
) -> ValueMap:
    result = Value(
        n_bands=n_bands,
        method="multiotsu",
        chroma=True,
        smooth=smooth > 0,
        flatten=False,
        seed=seed,
    ).extract(rgb)

    color = mean_color_by_value(rgb, result.labels)

    return ValueMap(
        image=result.scaffold,
        labels=result.labels,
        color=color,
        n_bands=result.n_bands,
    )
