from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ValueMethod = Literal["multiotsu", "kmeans", "percentile"]

DEFAULT_SEED = 7
DEFAULT_N_BANDS = 5

EPS = 1e-6

# approximate perceived-value for saturated colors.
# V = L + CHROMA_WEIGHT * C
CHROMA_WEIGHT = 0.16

# bilateral filter parameters
BILATERAL_SIGMA_COLOR = 0.10
BILATERAL_SIGMA_SPACE = 4.0

# multilevel otsu.
OTSU_N_BINS = 64

# illumination correction
ILLUM_SIGMA_FRAC = 0.08
ILLUM_STRENGTH = 0.50

# connected-component despeckling parameters
MIN_SPECKLE_AREA_PX = 64
SPECKLE_AREA_FRAC = 0.001


@dataclass
class ValueResult:
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    work: np.ndarray
    labels: np.ndarray
    scaffold: np.ndarray
    n_bands: int
    method: ValueMethod


@dataclass
class Value:
    n_bands: int = DEFAULT_N_BANDS
    method: ValueMethod = "multiotsu"
    chroma: bool = True
    smooth: bool = True
    flatten: bool = False
    seed: int = DEFAULT_SEED

    def extract(self, rgb: np.ndarray) -> ValueResult:
        check_rgb(rgb)

        L, C = lab_channels(rgb)
        V = perceived_value(L, C) if self.chroma else L

        work = V.copy()

        if self.flatten:
            work = flatten_illumination(work)

        if self.smooth:
            work = bilateral_smooth(work)

        y = self._quantize(work)
        y = sort_labels(y, work)

        y = despeckle(y)
        y = sort_labels(y, work)

        n = n_labels(y)
        scaffold = label_ramp(y, n)

        return ValueResult(
            L=L,
            C=C,
            V=V,
            work=work,
            labels=y,
            scaffold=scaffold,
            n_bands=n,
            method=self.method,
        )

    def _quantize(self, X: np.ndarray) -> np.ndarray:
        if self.method == "multiotsu":
            return quantize_multiotsu(X, self.n_bands)

        if self.method == "kmeans":
            return quantize_kmeans(X, self.n_bands, seed=self.seed)

        if self.method == "percentile":
            return quantize_percentile(X, self.n_bands)

        raise ValueError(f"unknown value method: {self.method!r}")


def check_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected RGB image with shape (H, W, 3), got {rgb.shape}")

    if not np.isfinite(rgb).all():
        raise ValueError("RGB image contains non-finite values")

    if rgb.min() < 0.0 or rgb.max() > 1.0:
        raise ValueError("RGB image must be normalized to [0, 1]")


def lab_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = rgb2lab(rgb)
    L = (lab[..., 0] / 100.0).astype("float32")
    C = (np.hypot(lab[..., 1], lab[..., 2]) / 100.0).astype("float32")
    return L, C


def perceived_value(
    L: np.ndarray,
    C: np.ndarray,
    chroma_weight: float = CHROMA_WEIGHT,
) -> np.ndarray:
    V = L + chroma_weight * C
    return np.clip(V, 0.0, 1.0).astype("float32")


def value_channel(
    rgb: np.ndarray,
    chroma_weight: float = CHROMA_WEIGHT,
) -> np.ndarray:
    L, C = lab_channels(rgb)
    return perceived_value(L, C, chroma_weight=chroma_weight)


def adjust_saturation(
    rgb: np.ndarray,
    saturation: float,
) -> np.ndarray:
    if abs(saturation - 1.0) < EPS:
        return rgb.astype("float32")

    hsv = cv2.cvtColor(rgb.astype("float32"), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)

    return np.clip(
        cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB),
        0.0,
        1.0,
    ).astype("float32")


def bilateral_smooth(
    X: np.ndarray,
    sigma_color: float = BILATERAL_SIGMA_COLOR,
    sigma_space: float = BILATERAL_SIGMA_SPACE,
) -> np.ndarray:
    d = int(2 * round(sigma_space) + 1)

    Y = cv2.bilateralFilter(
        X.astype("float32"),
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )

    return Y.astype("float32")


def flatten_illumination(
    X: np.ndarray,
    sigma_frac: float = ILLUM_SIGMA_FRAC,
    strength: float = ILLUM_STRENGTH,
) -> np.ndarray:
    sigma = max(X.shape) * sigma_frac

    B = cv2.GaussianBlur(
        X.astype("float32"),
        ksize=(0, 0),
        sigmaX=sigma,
    )

    Y = X - strength * (B - float(B.mean()))
    return np.clip(Y, 0.0, 1.0).astype("float32")


def quantize_multiotsu(
    X: np.ndarray,
    n_bands: int,
    n_bins: int = OTSU_N_BINS,
) -> np.ndarray:
    for k in range(max(2, n_bands), 1, -1):
        try:
            t = threshold_multiotsu(X, classes=k, nbins=n_bins)
        except ValueError:
            continue

        return np.digitize(X, bins=t).astype("int64")

    return np.zeros(X.shape, dtype="int64")


def kmeans_fit(
    X: np.ndarray,
    k: int,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=10,  # pyright: ignore[reportArgumentType]
    )

    with threadpool_limits(limits=1):
        labels = model.fit_predict(X)

    return labels, model.cluster_centers_


def quantize_kmeans(
    X: np.ndarray,
    n_bands: int,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    q = np.unique((X * 255).astype("uint8"))
    k = min(max(1, n_bands), len(q))

    if k <= 1:
        return np.zeros(X.shape, dtype="int64")

    labels, _ = kmeans_fit(X.reshape(-1, 1), k, seed)

    return labels.reshape(X.shape).astype("int64")


def quantize_percentile(
    X: np.ndarray,
    n_bands: int,
) -> np.ndarray:
    if n_bands <= 1:
        return np.zeros(X.shape, dtype="int64")

    p = np.linspace(0.0, 100.0, n_bands + 1)[1:-1]
    t = np.unique(np.percentile(X, p))

    if len(t) == 0:
        return np.zeros(X.shape, dtype="int64")

    return np.digitize(X, bins=t).astype("int64")


def sort_labels(
    labels: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    ids = np.unique(labels)

    mu = np.array(
        [X[labels == i].mean() for i in ids],
        dtype="float32",
    )

    order = ids[np.argsort(mu)]

    lut = np.zeros(int(ids.max()) + 1, dtype="int64")
    for new, old in enumerate(order):
        lut[old] = new

    return lut[labels].astype("int64")


def despeckle(
    labels: np.ndarray,
    area_min: int = MIN_SPECKLE_AREA_PX,
    area_frac: float = SPECKLE_AREA_FRAC,
    kernel: int = 7,
) -> np.ndarray:
    if kernel % 2 == 0:
        raise ValueError("kernel must be odd")

    area = max(area_min, int(labels.size * area_frac))
    fallback = cv2.medianBlur(labels.astype("uint8"), kernel)
    out = labels.copy()

    for k in np.unique(labels):
        cc = label(labels == k, connectivity=1)

        for reg in regionprops(cc):
            if reg.area >= area:
                continue

            rr = reg.coords[:, 0]
            col = reg.coords[:, 1]
            out[rr, col] = fallback[rr, col]

    return out.astype("int64")


def label_ramp(
    labels: np.ndarray,
    n_bands: int | None = None,
) -> np.ndarray:
    n = n_labels(labels) if n_bands is None else n_bands

    if n <= 1:
        return np.zeros(labels.shape, dtype="float32")

    ramp = np.linspace(0.0, 1.0, n, dtype="float32")
    return ramp[labels].astype("float32")


def n_labels(labels: np.ndarray) -> int:
    return int(labels.max()) + 1
