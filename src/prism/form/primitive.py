from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.segmentation import mark_boundaries, slic

from prism.color.value import Value, value_channel

SEED_DEFAULT = 7
EPS = 1e-6

WORK_SIZE = 200
N_RANDOM = 48
MAX_AGE = 24
EFFORT_LO = 0.45
MUT_SIGMA_FRAC = 0.06
SIZE_HI = 0.42
SIZE_LO = 0.05
SIZE_GAMMA = 1.5
MIN_FRAC_AXIS = 0.012
MIN_AREA_PX = 8
POLYGON_SIDES = 5


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < EPS:
        return np.zeros_like(x, dtype="float32")
    return ((x - lo) / (hi - lo)).astype("float32")


def preprocess_rgb(
    rgb: np.ndarray, denoise: float = 0.15, smooth: float = 0.25
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
        blur = cv2.GaussianBlur(out, ksize=(0, 0), sigmaX=max(0.5, 2.0 * smooth))
        out = (1 - smooth) * out + smooth * blur
    return np.clip(out, 0, 1).astype("float32")


@dataclass
class ValueBlock:
    image: np.ndarray
    labels: np.ndarray
    color: np.ndarray
    bands: int


def color_from_value_labels(
    rgb: np.ndarray,
    labels: np.ndarray,
    mode: Literal["mean"] = "mean",
) -> np.ndarray:
    if mode != "mean":
        raise ValueError(f"unsupported color mode: {mode!r}")
    flat = labels.ravel()
    n = int(flat.max()) + 1
    counts = np.maximum(np.bincount(flat, minlength=n), 1)[:, None]
    means = (
        np.stack(
            [
                np.bincount(
                    flat, weights=rgb[..., c].ravel().astype("float64"), minlength=n
                )
                for c in range(3)
            ],
            axis=1,
        )
        / counts
    )
    return np.asarray(means[labels], dtype="float32")


def value_block_in(
    rgb: np.ndarray,
    bands: int = 5,
    smooth: float = 0.25,
) -> ValueBlock:
    result = Value(
        n_bands=bands,
        method="multiotsu",
        chroma=True,
        smooth=smooth > 0,
        flatten=False,
        seed=SEED_DEFAULT,
    ).extract(rgb)
    color = color_from_value_labels(rgb, result.labels)
    return ValueBlock(
        image=result.scaffold, labels=result.labels, color=color, bands=result.n_bands
    )


@dataclass
class SuperpixelBlock:
    image: np.ndarray
    labels: np.ndarray


def superpixel_labels(
    rgb: np.ndarray,
    n_segments: int,
    compactness: float,
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


def superpixel_mean_value(labels: np.ndarray, value: np.ndarray) -> np.ndarray:
    flat = labels.ravel()
    n = int(flat.max()) + 1
    sums = np.bincount(flat, weights=value.ravel().astype("float64"), minlength=n)
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    return np.asarray((sums / counts)[labels], dtype="float32")


def superpixel_mean_color(labels: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    flat = labels.ravel()
    n = int(flat.max()) + 1
    counts = np.maximum(np.bincount(flat, minlength=n), 1)[:, None]
    means = (
        np.stack(
            [
                np.bincount(
                    flat, weights=rgb[..., c].ravel().astype("float64"), minlength=n
                )
                for c in range(3)
            ],
            axis=1,
        )
        / counts
    )
    return np.asarray(means[labels], dtype="float32")


def superpixel_block_in(
    rgb: np.ndarray,
    mode: Literal["boundaries", "value", "color"] = "value",
    n_segments: int = 600,
    compactness: float = 10,
    smooth: float = 1.0,
) -> SuperpixelBlock:
    labels = superpixel_labels(rgb, n_segments, compactness, smooth)
    if mode == "boundaries":
        image = np.asarray(
            mark_boundaries(rgb, labels, color=(1, 0, 0)), dtype="float32"
        )
    elif mode == "value":
        image = superpixel_mean_value(labels, value_channel(rgb))
    elif mode == "color":
        image = superpixel_mean_color(labels, rgb)
    else:
        raise ValueError(f"unknown superpixel mode: {mode!r}")
    return SuperpixelBlock(image=image, labels=labels)


ShapeFamily = Literal["ellipse", "rectangle", "triangle", "polygon", "combo"]
_SHAPES = ("ellipse", "rectangle", "triangle", "polygon")


@dataclass
class Shape:
    kind: str
    geom: tuple[float, ...]
    fill: float | tuple[float, ...]
    alpha: float
    score: float


@dataclass
class PrimitiveFit:
    image: np.ndarray
    residual: np.ndarray
    shapes: list[Shape]
    snapshots: dict[int, np.ndarray]


def edge_guide(target: np.ndarray) -> np.ndarray:
    gray = target if target.ndim == 2 else target.mean(axis=-1)
    gx = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
    return norm01(np.hypot(gx, gy))


def residual_guide(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    err = (target - current) ** 2
    if err.ndim == 3:
        err = err.mean(axis=-1)
    return norm01(err)


def _guide_cdf(guide: np.ndarray) -> np.ndarray:
    flat = guide.ravel().astype("float64")
    if flat.sum() < EPS:
        flat = np.ones_like(flat)
    return np.cumsum(flat / flat.sum())


def sample_center(
    rng: np.random.Generator, cdf: np.ndarray, w: int
) -> tuple[float, float]:
    idx = min(int(np.searchsorted(cdf, rng.random())), len(cdf) - 1)
    return float(idx % w), float(idx // w)


def _size_frac(i: int, n: int) -> float:
    return SIZE_HI + (SIZE_LO - SIZE_HI) * (i / max(n - 1, 1)) ** SIZE_GAMMA


def _search_budget(
    i: int, n: int, starts: int, effort_lo: float
) -> tuple[int, int, int]:
    f = effort_lo + (1.0 - effort_lo) * (1.0 - i / max(n - 1, 1))
    return (
        max(1, round(starts * f)),
        max(8, round(N_RANDOM * f)),
        max(8, round(MAX_AGE * f)),
    )


def propose_shape(
    rng: np.random.Generator,
    cx: float,
    cy: float,
    sf: float,
    diag: float,
    families: tuple[str, ...],
) -> tuple[str, tuple[float, ...]]:
    kind = families[int(rng.integers(len(families)))]
    lo, hi = MIN_FRAC_AXIS * diag, sf * diag
    if kind == "ellipse":
        return kind, (
            cx,
            cy,
            float(rng.uniform(lo, hi)),
            float(rng.uniform(lo, hi)),
            float(rng.uniform(0, 180)),
        )
    if kind == "rectangle":
        return kind, (
            cx,
            cy,
            float(rng.uniform(lo, hi)),
            float(rng.uniform(lo, hi)),
            float(rng.uniform(0, 180)),
        )
    if kind == "triangle":
        g = [c + float(rng.uniform(-hi, hi)) for _ in range(3) for c in (cx, cy)]
        return kind, tuple(g)
    base = float(rng.uniform(0, 2 * np.pi))
    g = []
    for k in range(POLYGON_SIDES):
        a = base + 2 * np.pi * k / POLYGON_SIDES + float(rng.uniform(-0.3, 0.3))
        rr = float(rng.uniform(lo, hi)) * float(rng.uniform(0.6, 1.0))
        g += [cx + rr * float(np.cos(a)), cy + rr * float(np.sin(a))]
    return kind, tuple(g)


def mutate_shape(
    rng: np.random.Generator,
    kind: str,
    geom: tuple[float, ...],
    diag: float,
) -> tuple[str, tuple[float, ...]]:
    g = list(geom)
    j = MUT_SIGMA_FRAC * diag
    if kind in ("ellipse", "rectangle"):
        g[0] += float(rng.normal(0, j))
        g[1] += float(rng.normal(0, j))
        g[2] = max(1.0, g[2] + float(rng.normal(0, j)))
        g[3] = max(1.0, g[3] + float(rng.normal(0, j)))
        g[4] = float((g[4] + rng.normal(0, 20)) % 180)
    else:
        k = int(rng.integers(len(g)))
        g[k] += float(rng.normal(0, j))
    return kind, tuple(g)


def _fill_poly(
    pts: np.ndarray,
    shape_hw: tuple[int, int],
    convex: bool,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape_hw
    x0, x1 = max(0, int(pts[:, 0].min())), min(w, int(pts[:, 0].max()) + 1)
    y0, y1 = max(0, int(pts[:, 1].min())), min(h, int(pts[:, 1].max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.empty(0, "int64"), np.empty(0, "int64")
    buf = np.zeros((y1 - y0, x1 - x0), "uint8")
    local = (pts - [x0, y0]).astype("int32")
    if convex:
        cv2.fillConvexPoly(buf, local, 1)
    else:
        cv2.fillPoly(buf, [local], 1)
    ys, xs = np.nonzero(buf)
    return ys + y0, xs + x0


def rasterize_shape(
    kind: str,
    geom: tuple[float, ...],
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape_hw
    if kind == "ellipse":
        cx, cy, ax, ay, ang = geom
        ax, ay = max(1.0, ax), max(1.0, ay)
        rad = int(np.ceil(max(ax, ay))) + 1
        x0, x1 = max(0, int(cx - rad)), min(w, int(cx + rad) + 1)
        y0, y1 = max(0, int(cy - rad)), min(h, int(cy + rad) + 1)
        if x1 <= x0 or y1 <= y0:
            return np.empty(0, "int64"), np.empty(0, "int64")
        buf = np.zeros((y1 - y0, x1 - x0), "uint8")
        cv2.ellipse(
            buf,
            (int(round(cx - x0)), int(round(cy - y0))),
            (int(round(ax)), int(round(ay))),
            ang,
            0,
            360,
            1,
            -1,
        )
        ys, xs = np.nonzero(buf)
        return ys + y0, xs + x0
    if kind == "rectangle":
        cx, cy, rw, rh, ang = geom
        pts = cv2.boxPoints(((cx, cy), (max(1.0, rw), max(1.0, rh)), ang))
        return _fill_poly(pts, shape_hw, convex=True)
    if kind == "triangle":
        return _fill_poly(
            np.array(geom, "float32").reshape(3, 2), shape_hw, convex=True
        )
    return _fill_poly(np.array(geom, "float32").reshape(-1, 2), shape_hw, convex=False)


def optimal_fill(
    target: np.ndarray,
    current: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    seg, cur = target[ys, xs], current[ys, xs]
    if alpha >= 1.0:
        return seg.mean(axis=0)
    return np.clip(cur.mean(axis=0) + (seg - cur).mean(axis=0) / alpha, 0.0, 1.0)


def score_candidate(
    target: np.ndarray,
    current: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    seg, cur = target[ys, xs], current[ys, xs]
    fill = optimal_fill(target, current, ys, xs, alpha)
    new = fill if alpha >= 1.0 else (1 - alpha) * cur + alpha * fill
    delta = float(((seg - new) ** 2).sum() - ((seg - cur) ** 2).sum())
    return delta, fill


def render_primitives(
    shapes: list[Shape],
    shape_hw: tuple[int, int],
    bg: float | np.ndarray,
    n_ch: int,
) -> np.ndarray:
    if n_ch == 1:
        out = np.full(shape_hw, float(bg), "float32")
    else:
        out = np.empty((*shape_hw, n_ch), "float32")
        out[:] = np.asarray(bg, "float32")
    for s in shapes:
        ys, xs = rasterize_shape(s.kind, s.geom, shape_hw)
        if len(ys) == 0:
            continue
        fill = np.asarray(s.fill, dtype="float32")
        out[ys, xs] = (
            fill if s.alpha >= 1.0 else (1 - s.alpha) * out[ys, xs] + s.alpha * fill
        )
    return np.clip(out, 0, 1).astype("float32")


def _scaled_geom(kind: str, geom: tuple[float, ...], s: float) -> tuple[float, ...]:
    if kind in ("ellipse", "rectangle"):
        cx, cy, a, b, ang = geom
        return (cx * s, cy * s, a * s, b * s, ang)
    return tuple(v * s for v in geom)


def _eval(target, current, kind, geom, wh, alpha):
    ys, xs = rasterize_shape(kind, geom, wh)
    if len(ys) < MIN_AREA_PX:
        return None
    delta, fill = score_candidate(target, current, ys, xs, alpha)
    return delta, fill, ys, xs, kind, geom


def _best_shape(
    rng,
    target,
    current,
    cdf,
    w,
    sf,
    diag,
    alpha,
    wh,
    families,
    restarts,
    n_random,
    max_age,
):
    best = None
    for _ in range(restarts):
        seed = None
        for _ in range(n_random):
            cx, cy = sample_center(rng, cdf, w)
            kind, geom = propose_shape(rng, cx, cy, sf, diag, families)
            r = _eval(target, current, kind, geom, wh, alpha)
            if r is not None and (seed is None or r[0] < seed[0]):
                seed = r
        if seed is None:
            continue
        cur, age = seed, 0
        while age < max_age:
            kind, geom = mutate_shape(rng, cur[4], cur[5], diag)
            r = _eval(target, current, kind, geom, wh, alpha)
            if r is not None and r[0] < cur[0]:
                cur, age = r, 0
            else:
                age += 1
        if best is None or cur[0] < best[0]:
            best = cur
    return best


def fit_primitives(
    target: np.ndarray,
    n_shapes: int = 120,
    shape_family: ShapeFamily = "combo",
    starts: int = 16,
    edge_weight: float = 0.5,
    residual_weight: float = 1.0,
    alpha: float = 0.6,
    work_size: int = WORK_SIZE,
    effort_lo: float = EFFORT_LO,
    snapshots: tuple[int, ...] = (10, 30, 80),
    seed: int = SEED_DEFAULT,
) -> PrimitiveFit:
    rng = np.random.default_rng(seed)
    h0, w0 = target.shape[:2]
    n_ch = 1 if target.ndim == 2 else target.shape[2]
    wh = (
        max(1, round(h0 * work_size / max(h0, w0))),
        max(1, round(w0 * work_size / max(h0, w0))),
    )
    t = cv2.resize(
        target.astype("float32"), (wh[1], wh[0]), interpolation=cv2.INTER_AREA
    )
    diag = float(np.hypot(*wh))
    scale_up = max(h0, w0) / max(wh)
    families = _SHAPES if shape_family == "combo" else (shape_family,)
    edge = edge_weight * edge_guide(t)

    bg = float(t.mean()) if n_ch == 1 else t.mean(axis=(0, 1))
    if n_ch == 1:
        current = np.full(wh, bg, "float32")
    else:
        current = np.empty((*wh, n_ch), "float32")
        current[:] = bg

    shapes: list[Shape] = []
    for i in range(n_shapes):
        guide = residual_weight * residual_guide(t, current) + edge
        cdf = _guide_cdf(guide)
        sf = _size_frac(i, n_shapes)
        restarts, n_random, max_age = _search_budget(i, n_shapes, starts, effort_lo)
        best = _best_shape(
            rng,
            t,
            current,
            cdf,
            wh[1],
            sf,
            diag,
            alpha,
            wh,
            families,
            restarts,
            n_random,
            max_age,
        )
        if best is None or best[0] >= 0:
            continue
        delta, fill, ys, xs, kind, geom = best
        current[ys, xs] = (
            fill if alpha >= 1.0 else (1 - alpha) * current[ys, xs] + alpha * fill
        )
        out_fill = (
            float(fill) if n_ch == 1 else tuple(float(v) for v in np.asarray(fill))
        )
        shapes.append(
            Shape(kind, _scaled_geom(kind, geom, scale_up), out_fill, alpha, delta)
        )

    image = render_primitives(shapes, (h0, w0), bg, n_ch)
    frames = {
        k: render_primitives(shapes[:k], (h0, w0), bg, n_ch)
        for k in snapshots
        if k <= len(shapes)
    }
    residual = np.abs(target - image)
    if residual.ndim == 3:
        residual = residual.mean(axis=-1)
    return PrimitiveFit(
        image=image,
        residual=residual.astype("float32"),
        shapes=shapes,
        snapshots=frames,
    )
