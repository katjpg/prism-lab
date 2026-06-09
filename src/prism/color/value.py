from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans


ValueMethod = Literal["multiotsu", "kmeans", "percentile"]

SEED_DEFAULT = 7
N_BANDS_DEFAULT = 5

EPS = 1e-6

# approximate perceived-value for saturated colors.
# V = L + W_CHROMA * C
W_CHROMA = 0.16

# bilateral filter parameters
SIGMA_COLOR = 0.10
SIGMA_SPACE = 4.0

# multilevel otsu.
N_BINS_OTSU = 64

# illumination correction
ILLUM_SIGMA_FRAC = 0.08
ILLUM_STRENGTH = 0.50

# connected-component despeckling parameters
SPECKLE_AREA_MIN = 64
SPECKLE_AREA_FRAC = 0.001


@dataclass
class ValueResult:
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    X: np.ndarray
    labels: np.ndarray
    scaffold: np.ndarray
    n_bands: int
    method: ValueMethod


@dataclass
class Value:
    n_bands: int = N_BANDS_DEFAULT
    method: ValueMethod = "multiotsu"
    chroma: bool = True
    smooth: bool = True
    flatten: bool = False
    seed: int = SEED_DEFAULT

    def extract(self, rgb: np.ndarray) -> ValueResult:
        check_rgb(rgb)

        L, C = lab_channels(rgb)
        V = perceived_value(L, C) if self.chroma else L

        X = V.copy()

        if self.flatten:
            X = flatten_illumination(X)

        if self.smooth:
            X = bilateral_smooth(X)

        y = self._quantize(X)
        y = sort_labels(y, X)

        y = despeckle(y)
        y = sort_labels(y, X)

        n = n_labels(y)
        scaffold = label2gray(y, n)

        return ValueResult(
            L=L,
            C=C,
            V=V,
            X=X,
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


def lab_luminance(rgb: np.ndarray) -> np.ndarray:
    lab = rgb2lab(rgb)
    return (lab[..., 0] / 100.0).astype("float32")


def lab_chroma(rgb: np.ndarray) -> np.ndarray:
    lab = rgb2lab(rgb)
    C = np.hypot(lab[..., 1], lab[..., 2])
    return (C / 100.0).astype("float32")


def perceived_value(
    L: np.ndarray,
    C: np.ndarray,
    w_chroma: float = W_CHROMA,
) -> np.ndarray:
    V = L + w_chroma * C
    return np.clip(V, 0.0, 1.0).astype("float32")


def value_channel(
    rgb: np.ndarray,
    w_chroma: float = W_CHROMA,
) -> np.ndarray:
    L, C = lab_channels(rgb)
    return perceived_value(L, C, w_chroma=w_chroma)


def bilateral_smooth(
    X: np.ndarray,
    sigma_color: float = SIGMA_COLOR,
    sigma_space: float = SIGMA_SPACE,
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
    n_bins: int = N_BINS_OTSU,
) -> np.ndarray:
    for k in range(max(2, n_bands), 1, -1):
        try:
            t = threshold_multiotsu(X, classes=k, nbins=n_bins)
        except ValueError:
            continue

        return np.digitize(X, bins=t).astype("int64")

    return np.zeros(X.shape, dtype="int64")


def quantize_kmeans(
    X: np.ndarray,
    n_bands: int,
    seed: int = SEED_DEFAULT,
) -> np.ndarray:
    q = np.unique((X * 255).astype("uint8"))
    k = min(max(1, n_bands), len(q))

    if k <= 1:
        return np.zeros(X.shape, dtype="int64")

    model = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=10,
    )

    y = model.fit_predict(X.reshape(-1, 1))
    return y.reshape(X.shape).astype("int64")


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
    area_min: int = SPECKLE_AREA_MIN,
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


def label2gray(
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
