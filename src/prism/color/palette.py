from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import lab2rgb, rgb2lab

from prism.color.value import DEFAULT_SEED, EPS, ValueResult, check_rgb, kmeans_fit
from prism.preset import Detail, fit_image

DEFAULT_N_COLORS = 8
DEFAULT_N_SAMPLES = 20_000

RECOLOR_SCHEMES: dict[str, list[tuple[float, float, float]]] = {
    "teal_orange": [
        (0.04, 0.09, 0.12),
        (0.18, 0.26, 0.30),
        (0.55, 0.50, 0.40),
        (0.92, 0.82, 0.68),
    ],
    "bleach": [
        (0.10, 0.11, 0.12),
        (0.33, 0.34, 0.35),
        (0.60, 0.60, 0.60),
        (0.88, 0.88, 0.88),
    ],
    "sepia": [
        (0.10, 0.07, 0.04),
        (0.32, 0.24, 0.16),
        (0.60, 0.48, 0.33),
        (0.92, 0.82, 0.66),
    ],
    "noir": [
        (0.03, 0.04, 0.07),
        (0.16, 0.19, 0.25),
        (0.42, 0.46, 0.52),
        (0.80, 0.83, 0.87),
    ],
    "warm_film": [
        (0.10, 0.07, 0.05),
        (0.32, 0.26, 0.20),
        (0.63, 0.55, 0.44),
        (0.94, 0.87, 0.74),
    ],
    "cool_film": [
        (0.05, 0.08, 0.12),
        (0.19, 0.25, 0.31),
        (0.46, 0.53, 0.60),
        (0.85, 0.89, 0.94),
    ],
}


@dataclass
class PaletteColor:
    rgb: tuple[float, float, float]
    lab: tuple[float, float, float]
    weight: float


@dataclass
class Palette:
    colors: list[PaletteColor]

    @classmethod
    def extract(
        cls,
        rgb: np.ndarray,
        n_colors: int = DEFAULT_N_COLORS,
        n_samples: int = DEFAULT_N_SAMPLES,
        seed: int = DEFAULT_SEED,
        detail: Detail = "standard",
        pixels: tuple[int, int] | None = None,
    ) -> Palette:
        check_rgb(rgb)

        rgb, _ = fit_image(rgb, detail, pixels)

        X = rgb2lab(rgb).reshape(-1, 3)
        Xs = sample_rows(X, n=n_samples, seed=seed)

        k = min(n_colors, len(Xs))

        y, centers = kmeans_fit(Xs, k, seed)
        counts = np.bincount(y, minlength=k)
        weights = counts / max(int(counts.sum()), 1)

        colors: list[PaletteColor] = []

        for lab, w in zip(centers, weights):
            rgb_ = lab2rgb(lab.reshape(1, 1, 3)).reshape(3)
            rgb_ = np.clip(rgb_, 0.0, 1.0)

            colors.append(
                PaletteColor(
                    rgb=(float(rgb_[0]), float(rgb_[1]), float(rgb_[2])),
                    lab=(float(lab[0]), float(lab[1]), float(lab[2])),
                    weight=float(w),
                )
            )

        colors.sort(key=lambda c: c.lab[0])
        return cls(colors=colors)

    def by_lightness(self) -> list[PaletteColor]:
        return sorted(self.colors, key=lambda c: c.lab[0])

    def by_weight(self) -> list[PaletteColor]:
        return sorted(self.colors, key=lambda c: c.weight, reverse=True)


def sample_rows(
    X: np.ndarray,
    n: int,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    if len(X) <= n:
        return X

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)

    return X[idx]


def lch_ramp(
    anchors: list[tuple[float, float, float]],
    n: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(anchors, dtype="float32").reshape(1, -1, 3)
    lab = rgb2lab(rgb).reshape(-1, 3)

    C = np.hypot(lab[:, 1], lab[:, 2])
    h = np.unwrap(np.arctan2(lab[:, 2], lab[:, 1]))

    xp = np.linspace(0.0, 1.0, len(anchors))
    x = np.linspace(0.0, 1.0, n)

    C_i = np.interp(x, xp, C)
    h_i = np.interp(x, xp, h)

    a = C_i * np.cos(h_i)
    b = C_i * np.sin(h_i)

    return a.astype("float32"), b.astype("float32")


def recolor(
    rgb: np.ndarray,
    value: ValueResult,
    anchors: list[tuple[float, float, float]],
    amount: float = 0.4,
    chroma: float = 1.2,
    sigma: float = 6.0,
) -> np.ndarray:
    check_rgb(rgb)

    a_lut, b_lut = lch_ramp(anchors)

    lo, hi = np.percentile(value.L, (2, 98))
    x = (value.L - lo) / max(float(hi - lo), EPS)
    x = np.clip(x, 0.0, 1.0)

    idx = (x * (len(a_lut) - 1)).astype("int64")

    a_t = a_lut[idx] * chroma
    b_t = b_lut[idx] * chroma

    if sigma > 0:
        a_t = cv2.GaussianBlur(a_t, ksize=(0, 0), sigmaX=sigma)
        b_t = cv2.GaussianBlur(b_t, ksize=(0, 0), sigmaX=sigma)

    lab = rgb2lab(rgb)

    # preserve original Lab L*.
    lab[..., 1] = (1.0 - amount) * lab[..., 1] + amount * a_t
    lab[..., 2] = (1.0 - amount) * lab[..., 2] + amount * b_t

    return np.clip(lab2rgb(lab), 0.0, 1.0).astype("float32")
