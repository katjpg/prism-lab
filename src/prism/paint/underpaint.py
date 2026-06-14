from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import lab2rgb, rgb2lab

from prism.color.value import value_channel

EDGE_AMOUNT = 0.8
EDGE_WIDTH = 1


@dataclass(frozen=True)
class Underpaint:
    shadow: tuple[float, float, float] = (0.12, 0.07, 0.04)
    highlight: tuple[float, float, float] = (0.93, 0.85, 0.66)
    gamma: float = 1.1
    chroma: float = 1.0


PIGMENTS = {
    "ultramarine": Underpaint(
        shadow=(0.05, 0.07, 0.26), highlight=(0.54, 0.62, 0.86), gamma=1.05, chroma=1.25
    ),
    "transparent_red_oxide": Underpaint(
        shadow=(0.15, 0.06, 0.03), highlight=(0.78, 0.50, 0.30), gamma=1.00, chroma=1.20
    ),
    "india_red": Underpaint(
        shadow=(0.13, 0.055, 0.060),
        highlight=(0.70, 0.46, 0.44),
        gamma=1.00,
        chroma=1.05,
    ),
    "raw_umber": Underpaint(
        shadow=(0.065, 0.060, 0.042),
        highlight=(0.60, 0.54, 0.42),
        gamma=1.05,
        chroma=0.90,
    ),
    "burnt_sienna": Underpaint(
        shadow=(0.13, 0.045, 0.015),
        highlight=(0.82, 0.47, 0.22),
        gamma=1.00,
        chroma=1.35,
    ),
}


@dataclass(frozen=True, slots=True)
class Underpainting:
    """A configurable underpainting ground beneath an acrylic/gouache layer.

    The ground maps the source's values onto a pigment tone, darkens edges, and
    blocks it in with a soft brush. Pass a pigment name for a preset, or a
    custom ``tone``; tune the stamp, pass, and edge darkening to taste.

    Parameters
    ----------
    pigment : str
        Name of a built-in pigment (a key of ``PIGMENTS``). Ignored when
        ``tone`` is given.
    tone : Underpaint or None, optional
        Custom shadow/highlight/gamma/chroma tone overriding ``pigment``.
    softness : float
        Anti-alias width of the soft block-in stamp; larger is smoother.
    blur : float
        Gaussian blur applied to the soft stamp.
    coverage : float
        Stroke size of the block-in pass.
    alpha : float
        Opacity of the block-in pass.
    edge_amount : float
        Strength of edge darkening, in ``0..1``.
    edge_width : int
        Dilation width of the darkened edges, in pixels.
    """

    pigment: str = "burnt_sienna"
    tone: Underpaint | None = None
    softness: float = 2.0
    blur: float = 0.0
    coverage: float = 1.6
    alpha: float = 0.85
    edge_amount: float = EDGE_AMOUNT
    edge_width: int = EDGE_WIDTH

    def spec(self) -> Underpaint:
        """Resolve the configured pigment or custom tone to an ``Underpaint``."""
        if self.tone is not None:
            return self.tone
        try:
            return PIGMENTS[self.pigment]
        except KeyError as exc:
            raise ValueError(f"unknown pigment: {self.pigment!r}") from exc


def value_to_underpaint(rgb: np.ndarray, spec: Underpaint) -> np.ndarray:
    v = value_channel(rgb)
    lo, hi = np.percentile(v, (2.0, 98.0))
    v = np.clip((v - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0) ** spec.gamma

    sh = rgb2lab(np.asarray(spec.shadow, np.float32).reshape(1, 1, 3)).reshape(3)
    hl = rgb2lab(np.asarray(spec.highlight, np.float32).reshape(1, 1, 3)).reshape(3)

    L = sh[0] + (hl[0] - sh[0]) * v
    a = (sh[1] + (hl[1] - sh[1]) * v) * spec.chroma
    b = (sh[2] + (hl[2] - sh[2]) * v) * spec.chroma

    lab = np.stack([L, a, b], axis=-1).astype(np.float32)
    return np.clip(lab2rgb(lab), 0.0, 1.0).astype(np.float32)


def edge_map(
    rgb: np.ndarray,
    sigma: float = 1.5,
    gamma: float = 0.6,
    width: int = 2,
) -> np.ndarray:
    g = value_channel(rgb)

    if sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    e = np.hypot(gx, gy)
    e = (e / max(float(e.max()), 1e-6)) ** gamma

    if width > 0:
        kernel = np.ones((2 * width + 1, 2 * width + 1), np.float32)
        e = cv2.dilate(e, kernel)

    return e.astype(np.float32)


def darken_edges(
    tonemap: np.ndarray,
    rgb: np.ndarray,
    amount: float = 0.6,
    width: int = 2,
) -> np.ndarray:
    if amount <= 0:
        return tonemap

    edge = edge_map(rgb, width=width)[..., None]
    return np.clip(tonemap * (1.0 - amount * edge), 0.0, 1.0).astype(np.float32)
