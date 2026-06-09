from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


ShapeFamily = Literal[
    "ellipse",
    "rectangle",
    "triangle",
    "polygon",
    "combo",
]

_SHAPE_PRESETS: dict[str, tuple[str, ...]] = {
    "ellipse": ("ellipse",),
    "rectangle": ("rectangle",),
    "triangle": ("triangle",),
    "polygon": ("polygon",),
    "combo": ("ellipse", "rectangle", "triangle", "polygon"),
}

SEED_DEFAULT = 7
EPS = 1e-6

WORK_SIZE = 200

N_RANDOM = 48
MAX_AGE = 24
EFFORT_LO = 0.45

MUT_SIGMA_FRAC = 0.06
SIZE_HI = 0.42
SIZE_LO = 0.035
SIZE_GAMMA = 1.5
MIN_FRAC_AXIS = 0.008
MIN_AREA_PX = 8
POLYGON_SIDES = 5

GUIDE_INTERVAL = 4
GUIDE_POOL = 4096
MIN_IMPROVEMENT = 1e-7
PATIENCE = 12

_EMPTY = np.empty(0, dtype="int64")


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


def fit_primitives(
    target: np.ndarray,
    n_shapes: int = 120,
    shape_family: ShapeFamily = "combo",
    starts: int = 8,
    edge_weight: float = 0.5,
    residual_weight: float = 1.0,
    alpha: float = 0.6,
    work_size: int = WORK_SIZE,
    effort_lo: float = EFFORT_LO,
    snapshots: tuple[int, ...] = (),
    seed: int = SEED_DEFAULT,
) -> PrimitiveFit:
    rng = np.random.default_rng(seed)

    h0, w0 = target.shape[:2]
    n_ch = 1 if target.ndim == 2 else target.shape[2]

    scale = work_size / max(h0, w0)
    h = max(1, round(h0 * scale))
    w = max(1, round(w0 * scale))
    shape_hw = (h, w)

    t = cv2.resize(
        target.astype("float32"),
        (w, h),
        interpolation=cv2.INTER_AREA,
    )

    diag = float(np.hypot(h, w))
    scale_up = max(h0, w0) / max(h, w)
    families = _families(shape_family)

    edge = edge_weight * edge_guide(t)

    bg = float(t.mean()) if n_ch == 1 else t.mean(axis=(0, 1))

    if n_ch == 1:
        current = np.full(shape_hw, bg, dtype="float32")
    else:
        current = np.empty((h, w, n_ch), dtype="float32")
        current[:] = bg

    shapes: list[Shape] = []
    pool = None
    stale = 0

    for i in range(n_shapes):
        if pool is None or i % _guide_interval(i, n_shapes) == 0:
            guide = residual_weight * residual_guide(t, current) + edge
            pool = guide_pool(guide, n=GUIDE_POOL)

        sf = _size_frac(i, n_shapes)
        restarts, n_random, max_age = _search_budget(
            i,
            n_shapes,
            starts=starts,
            effort_lo=effort_lo,
        )

        best = _best_shape(
            rng=rng,
            target=t,
            current=current,
            pool=pool,
            sf=sf,
            diag=diag,
            alpha=alpha,
            shape_hw=shape_hw,
            families=families,
            restarts=restarts,
            n_random=n_random,
            max_age=max_age,
        )

        if best is None or best[0] >= 0:
            stale += 1
            if stale >= PATIENCE:
                break
            continue

        score, fill, ys, xs, kind, geom = best
        improvement = -score

        if improvement < MIN_IMPROVEMENT:
            stale += 1
            if stale >= PATIENCE:
                break
        else:
            stale = 0

        _commit_shape(current, ys, xs, fill, alpha)

        out_fill = (
            float(fill) if n_ch == 1 else tuple(float(v) for v in np.asarray(fill))
        )

        shapes.append(
            Shape(
                kind=kind,
                geom=_scaled_geom(kind, geom, scale_up),
                fill=out_fill,
                alpha=alpha,
                score=score,
            )
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
        ys, xs = rasterize_shape(shape.kind, shape.geom, shape_hw)
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


def residual_guide(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    err = (target - current) ** 2

    if err.ndim == 3:
        err = err.mean(axis=-1)

    return norm01(err)


def guide_pool(
    guide: np.ndarray,
    n: int = GUIDE_POOL,
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


def propose_shape(
    rng: np.random.Generator,
    cx: float,
    cy: float,
    sf: float,
    diag: float,
    families: tuple[str, ...],
) -> tuple[str, tuple[float, ...]]:
    kind = families[int(rng.integers(len(families)))]
    lo = MIN_FRAC_AXIS * diag
    hi = sf * diag

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
        g: list[float] = []
        for _ in range(3):
            g.extend(
                [
                    cx + float(rng.uniform(-hi, hi)),
                    cy + float(rng.uniform(-hi, hi)),
                ]
            )
        return kind, tuple(g)

    base = float(rng.uniform(0, 2 * np.pi))
    g = []

    for k in range(POLYGON_SIDES):
        a = base + 2 * np.pi * k / POLYGON_SIDES + float(rng.uniform(-0.3, 0.3))
        r = float(rng.uniform(lo, hi)) * float(rng.uniform(0.6, 1.0))
        g.extend([cx + r * float(np.cos(a)), cy + r * float(np.sin(a))])

    return kind, tuple(g)


def mutate_shape(
    rng: np.random.Generator,
    kind: str,
    geom: tuple[float, ...],
    diag: float,
) -> tuple[str, tuple[float, ...]]:
    g = list(geom)
    step = MUT_SIGMA_FRAC * diag

    if kind in ("ellipse", "rectangle"):
        g[0] += float(rng.normal(0, step))
        g[1] += float(rng.normal(0, step))
        g[2] = max(1.0, g[2] + float(rng.normal(0, step)))
        g[3] = max(1.0, g[3] + float(rng.normal(0, step)))
        g[4] = float((g[4] + rng.normal(0, 20)) % 180)
    else:
        k = int(rng.integers(len(g)))
        g[k] += float(rng.normal(0, step))

    return kind, tuple(g)


def rasterize_shape(
    kind: str,
    geom: tuple[float, ...],
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape_hw

    if kind == "ellipse":
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

    if kind == "rectangle":
        cx, cy, rw, rh, angle = geom
        pts = cv2.boxPoints(((cx, cy), (max(1.0, rw), max(1.0, rh)), angle))
        return _fill_poly(pts, shape_hw, convex=True)

    if kind == "triangle":
        pts = np.asarray(geom, dtype="float32").reshape(3, 2)
        return _fill_poly(pts, shape_hw, convex=True)

    pts = np.asarray(geom, dtype="float32").reshape(-1, 2)
    return _fill_poly(pts, shape_hw, convex=False)


def optimal_fill(
    target: np.ndarray,
    current: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    seg = target[ys, xs]
    cur = current[ys, xs]

    if alpha >= 1.0:
        return seg.mean(axis=0)

    fill = cur.mean(axis=0) + (seg - cur).mean(axis=0) / alpha
    return np.clip(fill, 0.0, 1.0)


def score_shape(
    target: np.ndarray,
    current: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    seg = target[ys, xs]
    cur = current[ys, xs]
    fill = optimal_fill(target, current, ys, xs, alpha)

    if alpha >= 1.0:
        new = fill
    else:
        new = (1.0 - alpha) * cur + alpha * fill

    delta = float(((seg - new) ** 2).sum() - ((seg - cur) ** 2).sum())
    return delta, fill


def evaluate_shape(
    target: np.ndarray,
    current: np.ndarray,
    kind: str,
    geom: tuple[float, ...],
    shape_hw: tuple[int, int],
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, str, tuple[float, ...]] | None:
    ys, xs = rasterize_shape(kind, geom, shape_hw)

    if len(ys) < MIN_AREA_PX:
        return None

    score, fill = score_shape(target, current, ys, xs, alpha)

    return score, fill, ys, xs, kind, geom


def _best_shape(
    rng: np.random.Generator,
    target: np.ndarray,
    current: np.ndarray,
    pool: tuple[np.ndarray, np.ndarray, np.ndarray],
    sf: float,
    diag: float,
    alpha: float,
    shape_hw: tuple[int, int],
    families: tuple[str, ...],
    restarts: int,
    n_random: int,
    max_age: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, str, tuple[float, ...]] | None:
    best = None

    for _ in range(restarts):
        seed = None

        for _ in range(n_random):
            cx, cy = sample_center(rng, pool)
            kind, geom = propose_shape(rng, cx, cy, sf, diag, families)
            result = evaluate_shape(target, current, kind, geom, shape_hw, alpha)

            if result is not None and (seed is None or result[0] < seed[0]):
                seed = result

        if seed is None:
            continue

        cur = seed
        age = 0

        while age < max_age:
            kind, geom = mutate_shape(rng, cur[4], cur[5], diag)
            result = evaluate_shape(target, current, kind, geom, shape_hw, alpha)

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


def _commit_shape(
    current: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    fill: np.ndarray,
    alpha: float,
) -> None:
    if alpha >= 1.0:
        current[ys, xs] = fill
    else:
        current[ys, xs] = (1.0 - alpha) * current[ys, xs] + alpha * fill


def _scaled_geom(
    kind: str,
    geom: tuple[float, ...],
    scale: float,
) -> tuple[float, ...]:
    if kind in ("ellipse", "rectangle"):
        cx, cy, a, b, angle = geom
        return (cx * scale, cy * scale, a * scale, b * scale, angle)

    return tuple(v * scale for v in geom)


def _families(shape_family: ShapeFamily) -> tuple[str, ...]:
    try:
        return _SHAPE_PRESETS[shape_family]
    except KeyError as exc:
        raise ValueError(f"unknown shape family: {shape_family!r}") from exc


def _size_frac(i: int, n: int) -> float:
    return SIZE_HI + (SIZE_LO - SIZE_HI) * (i / max(n - 1, 1)) ** SIZE_GAMMA


def _search_budget(
    i: int,
    n: int,
    starts: int,
    effort_lo: float,
) -> tuple[int, int, int]:
    f = effort_lo + (1.0 - effort_lo) * (1.0 - i / max(n - 1, 1))

    return (
        max(1, round(starts * f)),
        max(8, round(N_RANDOM * f)),
        max(8, round(MAX_AGE * f)),
    )


def _guide_interval(i: int, n: int) -> int:
    if i >= int(n * 0.75):
        return 1

    if i >= int(n * 0.50):
        return 2

    return GUIDE_INTERVAL
