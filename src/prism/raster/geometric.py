from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numba import njit

from prism.color.value import DEFAULT_SEED
from prism.preset import Detail, fit_pixels

ShapeName = Literal[
    "ellipse",
    "rectangle",
    "triangle",
    "polygon",
]

ShapeFamily = Literal[
    "ellipse",
    "rectangle",
    "triangle",
    "polygon",
    "combo",
]

ShapeGeom: TypeAlias = tuple[float, ...]
EvalResult: TypeAlias = tuple[
    float, float | np.ndarray, np.ndarray, np.ndarray, ShapeName, ShapeGeom
]

_SHAPE_PRESETS: dict[ShapeFamily, tuple[ShapeName, ...]] = {
    "ellipse": ("ellipse",),
    "rectangle": ("rectangle",),
    "triangle": ("triangle",),
    "polygon": ("polygon",),
    "combo": ("ellipse", "rectangle", "triangle", "polygon"),
}

EPS = 1e-6

GEOM_SIDE: dict[Detail, int] = {
    "draft": 200,
    "standard": 280,
    "high": 380,
    "ultra": 480,
}
SHAPE_SCALE: dict[Detail, float] = {
    "draft": 0.7,
    "standard": 1.4,
    "high": 2.2,
    "ultra": 3.2,
}

N_RANDOM_SAMPLES = 48
MAX_STALE_ITERS = 24
MIN_EFFORT_SCALE = 0.45

MUTATION_SIGMA_FRAC = 0.06
MAX_SIZE_FRAC = 0.42
MIN_SIZE_FRAC = 0.035
SIZE_SCHEDULE_GAMMA = 1.5
MIN_AXIS_FRAC = 0.008
MIN_AREA_PX = 8
N_POLYGON_SIDES = 5

GUIDE_REFRESH_INTERVAL = 4
GUIDE_POOL_SIZE = 4096
MIN_IMPROVEMENT = 1e-7
PATIENCE = 12

_EMPTY = np.empty(0, dtype="int64")


@dataclass
class Shape:
    name: ShapeName
    geom: ShapeGeom
    fill: float | tuple[float, ...]
    alpha: float
    score: float


@dataclass
class PrimitiveFit:
    image: np.ndarray
    shapes: list[Shape]


def fit_primitives(
    target: np.ndarray,
    n_shapes: int = 120,
    shape_family: ShapeFamily = "combo",
    detail: Detail = "standard",
    pixels: tuple[int, int] | None = None,
    restarts: int = 8,
    edge_weight: float = 0.5,
    residual_weight: float = 1.0,
    alpha: float = 0.6,
    min_effort: float = MIN_EFFORT_SCALE,
    seed: int = DEFAULT_SEED,
) -> PrimitiveFit:
    rng = np.random.default_rng(seed)

    if pixels is not None:
        target, _ = fit_pixels(target, pixels)

    n_shapes = max(1, round(n_shapes * SHAPE_SCALE[detail]))

    h0, w0 = target.shape[:2]
    n_ch = 1 if target.ndim == 2 else target.shape[2]

    scale = GEOM_SIDE[detail] / max(h0, w0)
    h = max(1, round(h0 * scale))
    w = max(1, round(w0 * scale))
    shape_hw = (h, w)

    target_s = cv2.resize(
        target.astype("float32"),
        (w, h),
        interpolation=cv2.INTER_AREA,
    )

    diag = float(np.hypot(h, w))
    scale_up = max(h0, w0) / max(h, w)
    families = _families(shape_family)

    edge = edge_weight * edge_guide(target_s)

    bg = float(target_s.mean()) if n_ch == 1 else target_s.mean(axis=(0, 1))

    if n_ch == 1:
        canvas = np.full(shape_hw, bg, dtype="float32")
    else:
        canvas = np.empty((h, w, n_ch), dtype="float32")
        canvas[:] = bg

    shapes: list[Shape] = []
    pool = None
    stale = 0

    for i in range(n_shapes):
        if pool is None or i % _guide_interval(i, n_shapes) == 0:
            guide = residual_weight * residual_guide(target_s, canvas) + edge
            pool = guide_pool(guide, n=GUIDE_POOL_SIZE)

        sf = _size_frac(i, n_shapes)
        restart_budget, n_random, max_age = _search_budget(
            i,
            n_shapes,
            restarts=restarts,
            min_effort=min_effort,
        )

        best = _search_best(
            rng=rng,
            target=target_s,
            canvas=canvas,
            pool=pool,
            sf=sf,
            diag=diag,
            alpha=alpha,
            shape_hw=shape_hw,
            families=families,
            restarts=restart_budget,
            n_random=n_random,
            max_age=max_age,
        )

        if best is None or best[0] >= 0:
            stale += 1
            if stale >= PATIENCE:
                break
            continue

        score, fill, ys, xs, name, geom = best
        improvement = -score

        if improvement < MIN_IMPROVEMENT:
            stale += 1
            if stale >= PATIENCE:
                break
        else:
            stale = 0

        blend_shape_fill(canvas, ys, xs, fill, alpha)

        out_fill = float(fill) if n_ch == 1 else tuple(float(v) for v in np.asarray(fill))

        shapes.append(
            Shape(
                name=name,
                geom=_scaled_geom(name, geom, scale_up),
                fill=out_fill,
                alpha=alpha,
                score=score,
            )
        )

    image = render_primitives(shapes, (h0, w0), bg, n_ch)

    return PrimitiveFit(
        image=image,
        shapes=shapes,
    )


def render_primitives(
    shapes: list[Shape],
    shape_hw: tuple[int, int],
    bg: float | np.ndarray,
    n_ch: int,
) -> np.ndarray:
    if n_ch == 1:
        out = np.full(shape_hw, float(bg), dtype="float32")
    else:
        out = np.empty((*shape_hw, n_ch), dtype="float32")
        out[:] = np.asarray(bg, dtype="float32")

    for shape in shapes:
        ys, xs = rasterize_shape(shape.name, shape.geom, shape_hw)
        if len(ys) == 0:
            continue

        fill = np.asarray(shape.fill, dtype="float32")

        if shape.alpha >= 1.0:
            out[ys, xs] = fill
        else:
            out[ys, xs] = (1.0 - shape.alpha) * out[ys, xs] + shape.alpha * fill

    return np.clip(out, 0.0, 1.0).astype("float32")


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())

    if hi - lo < EPS:
        return np.zeros_like(x, dtype="float32")

    return ((x - lo) / (hi - lo)).astype("float32")


def edge_guide(target: np.ndarray) -> np.ndarray:
    gray = target if target.ndim == 2 else target.mean(axis=-1)

    gx = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)

    return norm01(np.hypot(gx, gy))


def residual_guide(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    err = (target - canvas) ** 2

    if err.ndim == 3:
        err = err.mean(axis=-1)

    return norm01(err)


def guide_pool(
    guide: np.ndarray,
    n: int = GUIDE_POOL_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = guide.ravel().astype("float64")
    n = min(n, flat.size)

    if float(flat.max()) <= EPS:
        idx = np.linspace(0, flat.size - 1, n, dtype="int64")
        weights = np.ones(n, dtype="float64")
    else:
        idx = np.argpartition(flat, -n)[-n:]
        weights = flat[idx]

    total = float(weights.sum())

    if total <= EPS:
        weights = np.ones_like(weights)
        total = float(weights.sum())

    ys, xs = np.divmod(idx, guide.shape[1])
    cdf = np.cumsum(weights / total)

    return ys.astype("int64"), xs.astype("int64"), cdf


def sample_center(
    rng: np.random.Generator,
    pool: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float]:
    ys, xs, cdf = pool
    i = min(int(np.searchsorted(cdf, rng.random())), len(cdf) - 1)

    return float(xs[i]), float(ys[i])


def sample_shape(
    rng: np.random.Generator,
    center_x: float,
    center_y: float,
    size_frac: float,
    diag: float,
    families: tuple[ShapeName, ...],
) -> tuple[ShapeName, ShapeGeom]:
    name = families[int(rng.integers(len(families)))]
    lo = MIN_AXIS_FRAC * diag
    hi = size_frac * diag

    if name == "ellipse":
        return name, (
            center_x,
            center_y,
            float(rng.uniform(lo, hi)),
            float(rng.uniform(lo, hi)),
            float(rng.uniform(0, 180)),
        )

    if name == "rectangle":
        return name, (
            center_x,
            center_y,
            float(rng.uniform(lo, hi)),
            float(rng.uniform(lo, hi)),
            float(rng.uniform(0, 180)),
        )

    if name == "triangle":
        geom_points: list[float] = []
        for _ in range(3):
            geom_points.extend(
                [
                    center_x + float(rng.uniform(-hi, hi)),
                    center_y + float(rng.uniform(-hi, hi)),
                ]
            )
        return name, tuple(geom_points)

    base = float(rng.uniform(0, 2 * np.pi))
    geom_points = []

    for k in range(N_POLYGON_SIDES):
        angle = base + 2 * np.pi * k / N_POLYGON_SIDES + float(rng.uniform(-0.3, 0.3))
        radius = float(rng.uniform(lo, hi)) * float(rng.uniform(0.6, 1.0))
        geom_points.extend(
            [
                center_x + radius * float(np.cos(angle)),
                center_y + radius * float(np.sin(angle)),
            ]
        )

    return name, tuple(geom_points)


def mutate_shape(
    rng: np.random.Generator,
    name: ShapeName,
    geom: ShapeGeom,
    diag: float,
) -> tuple[ShapeName, ShapeGeom]:
    geom_coords = list(geom)
    step = MUTATION_SIGMA_FRAC * diag

    if name in ("ellipse", "rectangle"):
        geom_coords[0] += float(rng.normal(0, step))
        geom_coords[1] += float(rng.normal(0, step))
        geom_coords[2] = max(1.0, geom_coords[2] + float(rng.normal(0, step)))
        geom_coords[3] = max(1.0, geom_coords[3] + float(rng.normal(0, step)))
        geom_coords[4] = float((geom_coords[4] + rng.normal(0, 20)) % 180)
    else:
        k = int(rng.integers(len(geom_coords)))
        geom_coords[k] += float(rng.normal(0, step))

    return name, tuple(geom_coords)


def rasterize_shape(
    name: ShapeName,
    geom: ShapeGeom,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape_hw

    if name == "ellipse":
        cx, cy, ax, ay, angle = geom
        ax = max(1.0, ax)
        ay = max(1.0, ay)

        rad = int(np.ceil(max(ax, ay))) + 1
        x0 = max(0, int(cx - rad))
        x1 = min(w, int(cx + rad) + 1)
        y0 = max(0, int(cy - rad))
        y1 = min(h, int(cy + rad) + 1)

        if x1 <= x0 or y1 <= y0:
            return _EMPTY, _EMPTY

        buf = np.zeros((y1 - y0, x1 - x0), dtype="uint8")

        cv2.ellipse(
            buf,
            (int(round(cx - x0)), int(round(cy - y0))),
            (int(round(ax)), int(round(ay))),
            angle,
            0,
            360,
            1,
            -1,
        )

        ys, xs = np.nonzero(buf)
        return ys + y0, xs + x0

    if name == "rectangle":
        cx, cy, rw, rh, angle = geom
        pts = cv2.boxPoints(((cx, cy), (max(1.0, rw), max(1.0, rh)), angle))
        return _fill_poly(pts, shape_hw, convex=True)

    if name == "triangle":
        pts = np.asarray(geom, dtype="float32").reshape(3, 2)
        return _fill_poly(pts, shape_hw, convex=True)

    pts = np.asarray(geom, dtype="float32").reshape(-1, 2)
    return _fill_poly(pts, shape_hw, convex=False)


@njit(cache=True, nogil=True)
def _score_gray(
    target: np.ndarray,
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    alpha: float,
):
    n = len(ys)

    t_sum = 0.0
    c_sum = 0.0
    for i in range(n):
        t_sum += target[ys[i], xs[i]]
        c_sum += canvas[ys[i], xs[i]]

    t_mean = t_sum / n
    c_mean = c_sum / n
    fill = t_mean if alpha >= 1.0 else c_mean + (t_mean - c_mean) / alpha
    if fill < 0.0:
        fill = 0.0
    elif fill > 1.0:
        fill = 1.0

    old_err = 0.0
    new_err = 0.0
    for i in range(n):
        t = target[ys[i], xs[i]]
        c = canvas[ys[i], xs[i]]
        new = fill if alpha >= 1.0 else (1.0 - alpha) * c + alpha * fill
        old_d = t - c
        new_d = t - new
        old_err += old_d * old_d
        new_err += new_d * new_d

    return new_err - old_err, fill


@njit(cache=True, nogil=True)
def _score_rgb(
    target: np.ndarray,
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    alpha: float,
):
    n = len(ys)

    t = np.zeros(3)
    c = np.zeros(3)
    for i in range(n):
        for k in range(3):
            t[k] += target[ys[i], xs[i], k]
            c[k] += canvas[ys[i], xs[i], k]

    fill = np.empty(3)
    for k in range(3):
        tm = t[k] / n
        cm = c[k] / n
        f = tm if alpha >= 1.0 else cm + (tm - cm) / alpha
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0
        fill[k] = f

    old_err = 0.0
    new_err = 0.0
    for i in range(n):
        for k in range(3):
            tv = target[ys[i], xs[i], k]
            cv = canvas[ys[i], xs[i], k]
            new = fill[k] if alpha >= 1.0 else (1.0 - alpha) * cv + alpha * fill[k]
            old_d = tv - cv
            new_d = tv - new
            old_err += old_d * old_d
            new_err += new_d * new_d

    return new_err - old_err, fill[0], fill[1], fill[2]


def score_shape(
    target: np.ndarray,
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    alpha: float,
) -> tuple[float, float | np.ndarray]:
    if target.ndim == 2:
        delta, fill = _score_gray(target, canvas, ys, xs, alpha)
        return float(delta), float(fill)

    delta, r, g, b = _score_rgb(target, canvas, ys, xs, alpha)
    return float(delta), np.array((r, g, b), dtype="float32")


def evaluate_shape(
    target: np.ndarray,
    canvas: np.ndarray,
    name: ShapeName,
    geom: ShapeGeom,
    shape_hw: tuple[int, int],
    alpha: float,
) -> EvalResult | None:
    ys, xs = rasterize_shape(name, geom, shape_hw)

    if len(ys) < MIN_AREA_PX:
        return None

    score, fill = score_shape(target, canvas, ys, xs, alpha)

    return score, fill, ys, xs, name, geom


def _search_best(
    rng: np.random.Generator,
    target: np.ndarray,
    canvas: np.ndarray,
    pool: tuple[np.ndarray, np.ndarray, np.ndarray],
    sf: float,
    diag: float,
    alpha: float,
    shape_hw: tuple[int, int],
    families: tuple[ShapeName, ...],
    restarts: int,
    n_random: int,
    max_age: int,
) -> EvalResult | None:
    best = None

    for _ in range(restarts):
        seed_shape = None

        for _ in range(n_random):
            cx, cy = sample_center(rng, pool)
            name, geom = sample_shape(rng, cx, cy, sf, diag, families)
            result = evaluate_shape(target, canvas, name, geom, shape_hw, alpha)

            if result is not None and (seed_shape is None or result[0] < seed_shape[0]):
                seed_shape = result

        if seed_shape is None:
            continue

        cur = seed_shape
        age = 0

        while age < max_age:
            name, geom = mutate_shape(rng, cur[4], cur[5], diag)
            result = evaluate_shape(target, canvas, name, geom, shape_hw, alpha)

            if result is not None and result[0] < cur[0]:
                cur = result
                age = 0
            else:
                age += 1

        if best is None or cur[0] < best[0]:
            best = cur

    return best


def _fill_poly(
    pts: np.ndarray,
    shape_hw: tuple[int, int],
    convex: bool,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape_hw

    x0 = max(0, int(pts[:, 0].min()))
    x1 = min(w, int(pts[:, 0].max()) + 1)
    y0 = max(0, int(pts[:, 1].min()))
    y1 = min(h, int(pts[:, 1].max()) + 1)

    if x1 <= x0 or y1 <= y0:
        return _EMPTY, _EMPTY

    buf = np.zeros((y1 - y0, x1 - x0), dtype="uint8")
    local = (pts - [x0, y0]).astype("int32")

    if convex:
        cv2.fillConvexPoly(buf, local, 1)
    else:
        cv2.fillPoly(buf, [local], 1)

    ys, xs = np.nonzero(buf)

    return ys + y0, xs + x0


@njit(cache=True, nogil=True)
def blend_fill_gray(
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    fill: float,
    alpha: float,
) -> None:
    for i in range(len(ys)):
        y = ys[i]
        x = xs[i]
        if alpha >= 1.0:
            canvas[y, x] = fill
        else:
            canvas[y, x] = (1.0 - alpha) * canvas[y, x] + alpha * fill


@njit(cache=True, nogil=True)
def blend_fill_rgb(
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    r: float,
    g: float,
    b: float,
    alpha: float,
) -> None:
    for i in range(len(ys)):
        y = ys[i]
        x = xs[i]
        if alpha >= 1.0:
            canvas[y, x, 0] = r
            canvas[y, x, 1] = g
            canvas[y, x, 2] = b
        else:
            canvas[y, x, 0] = (1.0 - alpha) * canvas[y, x, 0] + alpha * r
            canvas[y, x, 1] = (1.0 - alpha) * canvas[y, x, 1] + alpha * g
            canvas[y, x, 2] = (1.0 - alpha) * canvas[y, x, 2] + alpha * b


def blend_shape_fill(
    canvas: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    fill: float | np.ndarray,
    alpha: float,
) -> None:
    if canvas.ndim == 2:
        blend_fill_gray(canvas, ys, xs, float(fill), alpha)
    else:
        f = np.asarray(fill, dtype="float32")
        blend_fill_rgb(canvas, ys, xs, float(f[0]), float(f[1]), float(f[2]), alpha)


def _scaled_geom(
    name: ShapeName,
    geom: ShapeGeom,
    scale: float,
) -> tuple[float, ...]:
    if name in ("ellipse", "rectangle"):
        cx, cy, a, b, angle = geom
        return (cx * scale, cy * scale, a * scale, b * scale, angle)

    return tuple(v * scale for v in geom)


def _families(shape_family: ShapeFamily) -> tuple[ShapeName, ...]:
    try:
        return _SHAPE_PRESETS[shape_family]
    except KeyError as exc:
        raise ValueError(f"unknown shape family: {shape_family!r}") from exc


def _size_frac(i: int, n: int) -> float:
    return (
        MAX_SIZE_FRAC + (MIN_SIZE_FRAC - MAX_SIZE_FRAC) * (i / max(n - 1, 1)) ** SIZE_SCHEDULE_GAMMA
    )


def _search_budget(
    i: int,
    n: int,
    restarts: int,
    min_effort: float,
) -> tuple[int, int, int]:
    f = min_effort + (1.0 - min_effort) * (1.0 - i / max(n - 1, 1))

    return (
        max(1, round(restarts * f)),
        max(8, round(N_RANDOM_SAMPLES * f)),
        max(8, round(MAX_STALE_ITERS * f)),
    )


def _guide_interval(i: int, n: int) -> int:
    if i >= int(n * 0.75):
        return 1

    if i >= int(n * 0.50):
        return 2

    return GUIDE_REFRESH_INTERVAL


__all__ = [
    "PrimitiveFit",
    "Shape",
    "ShapeFamily",
    "ShapeName",
    "fit_primitives",
    "render_primitives",
]
