from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

from prism.color.value import EPS, DEFAULT_SEED, ValueResult, check_rgb


PaletteTone = Literal["shadow", "midtone", "highlight"]

DEFAULT_N_COLORS = 8
DEFAULT_N_SAMPLES = 20_000

# L* is on a scale of 0-100, where black = 0 and white = 100
SHADOW_L_MAX = 33.0
HIGHLIGHT_L_MIN = 66.0


@dataclass
class PaletteColor:
    rgb: tuple[float, float, float]
    lab: tuple[float, float, float]
    weight: float

    @property
    def tone(self) -> PaletteTone:
        L = self.lab[0]
        if L < SHADOW_L_MAX:
            return "shadow"
        if L < HIGHLIGHT_L_MIN:
            return "midtone"
        return "highlight"


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
    ) -> Palette:
        check_rgb(rgb)

        X = rgb2lab(rgb).reshape(-1, 3)
        Xs = sample_rows(X, n=n_samples, seed=seed)

        k = min(n_colors, len(Xs))

        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10,
        )

        y = model.fit_predict(Xs)
        counts = np.bincount(y, minlength=k)
        weights = counts / max(int(counts.sum()), 1)

        colors: list[PaletteColor] = []

        for lab, w in zip(model.cluster_centers_, weights):
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

    def by_tone(self, tone: PaletteTone) -> list[PaletteColor]:
        return [c for c in self.colors if c.tone == tone]

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
    tone: float = 0.4,
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
    lab[..., 1] = (1.0 - tone) * lab[..., 1] + tone * a_t
    lab[..., 2] = (1.0 - tone) * lab[..., 2] + tone * b_t

    return np.clip(lab2rgb(lab), 0.0, 1.0).astype("float32")


def swatch(
    palette: Palette,
    tone: PaletteTone | None = None,
) -> np.ndarray:
    colors = palette.colors if tone is None else palette.by_tone(tone)
    colors = sorted(colors, key=lambda c: c.lab[0])

    if not colors:
        return np.zeros((1, 1, 3), dtype="float32")

    bar = np.zeros((1, len(colors), 3), dtype="float32")

    for i, c in enumerate(colors):
        bar[0, i] = c.rgb

    return bar
