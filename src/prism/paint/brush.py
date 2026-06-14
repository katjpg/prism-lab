from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numba import njit, prange

BrushShape = Literal["round", "flat", "filbert", "bright", "fan", "split", "angular"]
RenderMode = Literal["splat", "line"]


EPS = 1e-8

DEFAULT_SEED = 7
DEFAULT_ARCLENGTH_STEP = 1.5
DEFAULT_BEZIER_SAMPLES = 12
DEFAULT_RADIUS = 12.0
DEFAULT_N_BRISTLES = 48
DEFAULT_DRAG = 0.08
DEFAULT_GRAIN_DEPTH = 0.18
DEFAULT_GRAIN_SCALE = 0.12


@dataclass
class Brush:
    xy: np.ndarray
    L: np.ndarray
    rho: np.ndarray
    r: np.ndarray
    a: np.ndarray
    eta: np.ndarray
    shape: str


def create_brush(
    radius: float = DEFAULT_RADIUS,
    n_bristles: int = DEFAULT_N_BRISTLES,
    shape: BrushShape = "round",
    seed: int = DEFAULT_SEED,
    L_rng: tuple[float, float] = (1.2, 2.2),
    rho_rng: tuple[float, float] = (0.10, 0.28),
    r_rng: tuple[float, float] = (0.7, 1.8),
    a_rng: tuple[float, float] = (0.55, 1.00),
    eta_rng: tuple[float, float] = (0.02, 0.16),
) -> Brush:
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_bristles <= 0:
        raise ValueError("n_bristles must be positive")

    rng = np.random.default_rng(seed)

    xy = _bristle_offsets(radius=radius, n=n_bristles, shape=shape, rng=rng)

    L = radius * rng.uniform(L_rng[0], L_rng[1], size=n_bristles)
    rho = rng.uniform(rho_rng[0], rho_rng[1], size=n_bristles)
    r = rng.uniform(r_rng[0], r_rng[1], size=n_bristles)
    a = rng.uniform(a_rng[0], a_rng[1], size=n_bristles)
    eta = rng.uniform(eta_rng[0], eta_rng[1], size=n_bristles)

    return Brush(
        xy=xy.astype(np.float32),
        L=L.astype(np.float32),
        rho=rho.astype(np.float32),
        r=r.astype(np.float32),
        a=a.astype(np.float32),
        eta=eta.astype(np.float32),
        shape=str(shape),
    )


def rasterize_stroke(
    shape_hw: tuple[int, int],
    path: np.ndarray,
    brush: Brush,
    samples: int = DEFAULT_BEZIER_SAMPLES,
    drag: float = DEFAULT_DRAG,
    grain_depth: float = DEFAULT_GRAIN_DEPTH,
    grain_scale: float = DEFAULT_GRAIN_SCALE,
    rotate: bool = False,
    render: RenderMode = "splat",
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    h, w = shape_hw
    vertices = np.asarray(path, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 2:
        return np.zeros((h, w), dtype=np.float32)

    direction = vertices[1] - vertices[0]
    if float(np.hypot(float(direction[0]), float(direction[1]))) < EPS:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    direction = direction / max(float(np.hypot(float(direction[0]), float(direction[1]))), EPS)

    n_bristles = brush.xy.shape[0]
    tip_offset = (-direction[None, :] * brush.L[:, None]).astype(np.float32)
    ctrl_offset = (0.5 * tip_offset).astype(np.float32)

    rng = np.random.default_rng(seed)
    n_seg = len(vertices) - 1
    noise = rng.uniform(0.0, 0.25, size=(n_seg, n_bristles)).astype(np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    render_code = 1 if render == "line" else 0
    rotate_code = 1 if rotate else 0

    _render_stroke_kernel(
        mask,
        vertices,
        brush.xy,
        brush.L,
        brush.rho,
        brush.r,
        brush.a,
        brush.eta,
        ctrl_offset,
        tip_offset,
        noise,
        np.float32(drag),
        int(samples),
        render_code,
        rotate_code,
    )

    if grain_depth > 0 or grain_scale > 0:
        noise_field = value_noise(
            mask.shape, seed=seed + 101, sigma=max(0.0, 0.75 + 5.0 * grain_scale)
        )
        mask = mask * (1.0 - grain_depth + grain_depth * noise_field)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def resample_path(path: np.ndarray, step: float = DEFAULT_ARCLENGTH_STEP) -> np.ndarray:
    vertices = np.asarray(path, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("path must have shape (N, 2)")
    if len(vertices) <= 1 or step <= 0:
        return vertices.copy()

    resampled = [vertices[0]]
    for start, end in zip(vertices[:-1], vertices[1:]):
        seg_len = float(np.hypot(*(end - start)))
        if seg_len < EPS:
            continue
        n_steps = max(1, int(np.ceil(seg_len / step)))
        t = np.linspace(1.0 / n_steps, 1.0, n_steps)[:, None]
        resampled.extend((1.0 - t) * start + t * end)

    return np.asarray(resampled, dtype=np.float32)


def value_noise(
    shape_hw: tuple[int, int],
    seed: int = DEFAULT_SEED,
    sigma: float = 1.5,
) -> np.ndarray:
    h, w = shape_hw
    rng = np.random.default_rng(seed)
    x = rng.random((h, w), dtype=np.float32)

    if sigma <= 0:
        return x

    y = cv2.GaussianBlur(x, (0, 0), sigmaX=float(sigma))
    y = y - y.min()
    den = max(float(y.max()), EPS)
    return (y / den).astype(np.float32)


def _bristle_offsets(
    radius: float,
    n: int,
    shape: BrushShape,
    rng: np.random.Generator,
) -> np.ndarray:
    if shape == "round":
        t = rng.uniform(0.0, 2.0 * np.pi, size=n)
        r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))
        return np.column_stack([r * np.cos(t), r * np.sin(t)])

    if shape == "flat":
        x = rng.uniform(-0.25 * radius, 0.25 * radius, size=n)
        y = rng.uniform(-1.00 * radius, 1.00 * radius, size=n)
        return np.column_stack([x, y])

    if shape == "filbert":
        t = rng.uniform(0.0, 2.0 * np.pi, size=n)
        rr = np.sqrt(rng.uniform(0.0, 1.0, size=n))
        x = 0.55 * radius * rr * np.cos(t)
        y = 1.00 * radius * rr * np.sin(t)
        return np.column_stack([x, y])

    if shape == "bright":
        x = rng.uniform(-0.45, 0.45, size=n) * radius
        y = rng.uniform(-0.50, 0.50, size=n) * radius
        return np.column_stack([x, y])

    if shape == "fan":
        u = rng.uniform(-1.0, 1.0, size=n)
        x = 0.95 * radius * u
        y = 0.10 * radius * rng.uniform(-1.0, 1.0, size=n) + 0.22 * radius * u * u
        return np.column_stack([x, y])

    if shape == "split":
        n_tufts = 5
        tuft = rng.integers(0, n_tufts, size=n)
        centers = np.linspace(-0.85, 0.85, n_tufts) * radius
        x = centers[tuft] + rng.normal(0.0, 0.05 * radius, size=n)
        y = rng.uniform(-1.0, 1.0, size=n) * radius
        return np.column_stack([x, y])

    if shape == "angular":
        x0 = rng.uniform(-0.22, 0.22, size=n) * radius
        y0 = rng.uniform(-1.0, 1.0, size=n) * radius
        c, s = np.cos(0.5), np.sin(0.5)
        x = x0 * c - y0 * s
        y = x0 * s + y0 * c
        return np.column_stack([x, y])

    raise ValueError(f"unknown brush shape: {shape}")


@njit(cache=True, fastmath=True, nogil=True)
def _normalize_xy(x: float, y: float) -> tuple[float, float]:
    n = np.sqrt(x * x + y * y)
    if n < EPS:
        return 0.0, 0.0
    return x / n, y / n


@njit(cache=True, fastmath=True, nogil=True)
def _update_bristles(
    ctrl_offset: np.ndarray,
    tip_offset: np.ndarray,
    direction: np.ndarray,
    length: np.ndarray,
    rho: np.ndarray,
    eta: np.ndarray,
    noise: np.ndarray,
    drag: np.float32,
):
    n = ctrl_offset.shape[0]
    next_ctrl = np.empty_like(ctrl_offset)
    next_tip = np.empty_like(tip_offset)

    dx = -float(direction[0])
    dy = -float(direction[1])
    tx, ty = _normalize_xy(dx, dy)

    for i in range(n):
        ux, uy = _normalize_xy(float(tip_offset[i, 0]), float(tip_offset[i, 1]))

        mx = float(eta[i]) * ux + (1.0 - float(eta[i])) * tx
        my = float(eta[i]) * uy + (1.0 - float(eta[i])) * ty
        mx, my = _normalize_xy(mx, my)

        t2x = float(length[i]) * mx
        t2y = float(length[i]) * my

        nx = float(tip_offset[i, 0]) + float(rho[i]) * (t2x - float(tip_offset[i, 0]))
        ny = float(tip_offset[i, 1]) + float(rho[i]) * (t2y - float(tip_offset[i, 1]))
        nx, ny = _normalize_xy(nx, ny)
        nx *= float(length[i])
        ny *= float(length[i])

        cx = 0.5 * nx
        cy = 0.5 * ny

        a = float(noise[i])
        cx = (1.0 - a) * cx + a * float(ctrl_offset[i, 0])
        cy = (1.0 - a) * cy + a * float(ctrl_offset[i, 1])

        next_ctrl[i, 0] = np.float32((1.0 - drag) * cx + drag * 0.5 * float(ctrl_offset[i, 0]))
        next_ctrl[i, 1] = np.float32((1.0 - drag) * cy + drag * 0.5 * float(ctrl_offset[i, 1]))
        next_tip[i, 0] = np.float32(nx)
        next_tip[i, 1] = np.float32(ny)

    return next_ctrl, next_tip


@njit(cache=True, fastmath=True, nogil=True)
def _accumulate_disc(
    accum: np.ndarray,
    x: float,
    y: float,
    radius: float,
    alpha: float,
) -> None:
    h, w = accum.shape
    rad = max(radius, 0.5)
    rad2 = rad * rad

    x_min = max(0, int(np.floor(x - rad)))
    x_max = min(w - 1, int(np.ceil(x + rad)))
    y_min = max(0, int(np.floor(y - rad)))
    y_max = min(h - 1, int(np.ceil(y + rad)))

    for yy in range(y_min, y_max + 1):
        dy = (yy + 0.5) - y
        for xx in range(x_min, x_max + 1):
            dx = (xx + 0.5) - x
            d2 = dx * dx + dy * dy
            if d2 > rad2:
                continue

            weight = 1.0 - d2 / (rad2 + EPS)
            value = accum[yy, xx] + alpha * weight
            accum[yy, xx] = 1.0 if value > 1.0 else value


@njit(cache=True, fastmath=True, nogil=True)
def _accumulate_line(
    accum: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    alpha: float,
) -> None:
    h, w = accum.shape
    ix0 = int(round(x0))
    iy0 = int(round(y0))
    ix1 = int(round(x1))
    iy1 = int(round(y1))

    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1
    err = dx - dy

    x = ix0
    y = iy0
    while True:
        if 0 <= x < w and 0 <= y < h:
            value = accum[y, x] + alpha
            accum[y, x] = 1.0 if value > 1.0 else value

        if x == ix1 and y == iy1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


@njit(cache=True, fastmath=True, nogil=True)
def _deposit_bristles(
    accum: np.ndarray,
    cx: float,
    cy: float,
    bristle_xy: np.ndarray,
    ctrl_offset: np.ndarray,
    tip_offset: np.ndarray,
    radius: np.ndarray,
    alpha: np.ndarray,
    samples: int,
    render_code: int,
) -> None:
    if samples < 2:
        samples = 2

    for i in range(bristle_xy.shape[0]):
        x0 = cx + float(bristle_xy[i, 0])
        y0 = cy + float(bristle_xy[i, 1])
        x1 = x0 + float(ctrl_offset[i, 0])
        y1 = y0 + float(ctrl_offset[i, 1])
        x2 = x0 + float(tip_offset[i, 0])
        y2 = y0 + float(tip_offset[i, 1])

        if render_code == 1:
            x_prev = x0
            y_prev = y0
            for k in range(1, samples + 1):
                t = k / samples
                s = 1.0 - t
                x = s * s * x0 + 2.0 * s * t * x1 + t * t * x2
                y = s * s * y0 + 2.0 * s * t * y1 + t * t * y2
                taper = 1.0 - 0.25 * t
                _accumulate_line(accum, x_prev, y_prev, x, y, float(alpha[i]) * taper)
                x_prev = x
                y_prev = y
        else:
            for k in range(samples + 1):
                t = k / samples
                s = 1.0 - t
                x = s * s * x0 + 2.0 * s * t * x1 + t * t * x2
                y = s * s * y0 + 2.0 * s * t * y1 + t * t * y2
                taper = 1.0 - 0.25 * t
                _accumulate_disc(accum, x, y, float(radius[i]), float(alpha[i]) * taper)


@njit(cache=True, fastmath=True, nogil=True)
def _render_stroke_kernel(
    accum: np.ndarray,
    path_xy: np.ndarray,
    bristle_xy: np.ndarray,
    length: np.ndarray,
    rho: np.ndarray,
    radius: np.ndarray,
    alpha: np.ndarray,
    eta: np.ndarray,
    ctrl_offset: np.ndarray,
    tip_offset: np.ndarray,
    noise: np.ndarray,
    drag: np.float32,
    samples: int,
    render_code: int,
    rotate_code: int,
) -> None:
    n_path = path_xy.shape[0]
    direction = np.empty(2, dtype=np.float32)
    rot_xy = np.empty_like(bristle_xy)
    seg_idx = 0

    for k in range(1, n_path):
        dx = path_xy[k, 0] - path_xy[k - 1, 0]
        dy = path_xy[k, 1] - path_xy[k - 1, 1]
        if np.sqrt(dx * dx + dy * dy) < EPS:
            continue

        direction[0] = dx
        direction[1] = dy

        ctrl_offset, tip_offset = _update_bristles(
            ctrl_offset, tip_offset, direction, length, rho, eta, noise[seg_idx], drag
        )

        if rotate_code == 1:
            ang = np.arctan2(np.float64(dy), np.float64(dx))
            ca = np.float32(np.cos(ang))
            sa = np.float32(np.sin(ang))
            for i in range(bristle_xy.shape[0]):
                rot_xy[i, 0] = ca * bristle_xy[i, 0] - sa * bristle_xy[i, 1]
                rot_xy[i, 1] = sa * bristle_xy[i, 0] + ca * bristle_xy[i, 1]
            step_xy = rot_xy
        else:
            step_xy = bristle_xy

        _deposit_bristles(
            accum,
            np.float64(path_xy[k, 0]),
            np.float64(path_xy[k, 1]),
            step_xy,
            ctrl_offset,
            tip_offset,
            radius,
            alpha,
            samples,
            render_code,
        )

        seg_idx += 1


DEFAULT_TILE = 16


@dataclass(frozen=True, slots=True)
class Stroke:
    alpha: np.ndarray
    color: np.ndarray
    top: int
    left: int


def place(center_y: float, center_x: float, cy: float, cx: float) -> tuple[int, int]:
    return int(round(center_y - cy)), int(round(center_x - cx))


@njit(cache=True, nogil=True, fastmath=False)
def composite_over_bbox(
    canvas: np.ndarray,
    alpha: np.ndarray,
    color: np.ndarray,
    top: int,
    left: int,
) -> None:
    H, W, C = canvas.shape
    h, w = alpha.shape
    y0 = 0 if top < 0 else top
    x0 = 0 if left < 0 else left
    y1 = top + h
    x1 = left + w
    if y1 > H:
        y1 = H
    if x1 > W:
        x1 = W
    if y0 >= y1 or x0 >= x1:
        return
    for y in range(y0, y1):
        ay = y - top
        for x in range(x0, x1):
            a = alpha[ay, x - left]
            if a <= 0.0:
                continue
            ia = 1.0 - a
            for c in range(C):
                canvas[y, x, c] = ia * canvas[y, x, c] + a * color[c]


def render_serial(canvas: np.ndarray, strokes: list[Stroke]) -> np.ndarray:
    for s in strokes:
        composite_over_bbox(canvas, s.alpha, np.ascontiguousarray(s.color), s.top, s.left)
    return canvas


@njit(cache=True, nogil=True, fastmath=False)
def _accumulate_coverage(
    coverage: np.ndarray,
    alpha: np.ndarray,
    top: int,
    left: int,
) -> None:
    H, W = coverage.shape
    h, w = alpha.shape
    y0 = 0 if top < 0 else top
    x0 = 0 if left < 0 else left
    y1 = top + h
    x1 = left + w
    if y1 > H:
        y1 = H
    if x1 > W:
        x1 = W
    if y0 >= y1 or x0 >= x1:
        return
    for y in range(y0, y1):
        ay = y - top
        for x in range(x0, x1):
            a = alpha[ay, x - left]
            if a <= 0.0:
                continue
            coverage[y, x] = a + (1.0 - a) * coverage[y, x]


def accumulate_coverage(coverage: np.ndarray, strokes: list[Stroke]) -> None:
    """Accumulate stroke coverage into ``coverage`` in place.

    Parameters
    ----------
    coverage : np.ndarray
        Coverage image to update, shape ``(H, W)``, float32, range ``0..1``.
    strokes : list[Stroke]
        Strokes whose alpha masks are accumulated.

    Notes
    -----
    Coverage is accumulated as ``1 - prod(1 - a_i)``, so the result does not
    depend on stroke order.
    """
    for s in strokes:
        _accumulate_coverage(coverage, s.alpha, s.top, s.left)


def render_coverage(shape_hw: tuple[int, int], strokes: list[Stroke]) -> np.ndarray:
    """Render stroke coverage into a single alpha image.

    Parameters
    ----------
    shape_hw : tuple[int, int]
        Output ``(height, width)``.
    strokes : list[Stroke]
        Strokes whose alpha masks are accumulated.

    Returns
    -------
    np.ndarray
        Coverage image, shape ``(height, width)``, float32, range ``0..1``.
    """
    coverage = np.zeros(shape_hw, dtype=np.float32)
    for s in strokes:
        _accumulate_coverage(coverage, s.alpha, s.top, s.left)
    return coverage


def pack_strokes(
    strokes: list[Stroke],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(strokes)
    tops = np.empty(n, np.int32)
    lefts = np.empty(n, np.int32)
    hs = np.empty(n, np.int32)
    ws = np.empty(n, np.int32)
    colors = np.empty((n, 3), np.float32)
    sizes = np.empty(n, np.int64)
    for i, s in enumerate(strokes):
        h, w = s.alpha.shape
        tops[i] = s.top
        lefts[i] = s.left
        hs[i] = h
        ws[i] = w
        colors[i] = s.color
        sizes[i] = h * w
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(sizes, out=offsets[1:])
    pool = np.empty(int(offsets[-1]), np.float32)
    for i, s in enumerate(strokes):
        pool[offsets[i] : offsets[i + 1]] = np.ascontiguousarray(s.alpha, np.float32).ravel()
    return tops, lefts, hs, ws, colors, pool, offsets


@njit(cache=True, nogil=True)
def bin_strokes_to_tiles(
    tops: np.ndarray,
    lefts: np.ndarray,
    hs: np.ndarray,
    ws: np.ndarray,
    H: int,
    W: int,
    tile: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    n = tops.size
    ntx = (W + tile - 1) // tile
    nty = (H + tile - 1) // tile
    nt = ntx * nty
    counts = np.zeros(nt, np.int64)
    for i in range(n):
        yy0 = tops[i]
        yy1 = tops[i] + hs[i] - 1
        xx0 = lefts[i]
        xx1 = lefts[i] + ws[i] - 1
        if yy0 < 0:
            yy0 = 0
        if xx0 < 0:
            xx0 = 0
        if yy1 > H - 1:
            yy1 = H - 1
        if xx1 > W - 1:
            xx1 = W - 1
        if yy0 > yy1 or xx0 > xx1:
            continue
        ty0 = yy0 // tile
        ty1 = yy1 // tile
        tx0 = xx0 // tile
        tx1 = xx1 // tile
        for ty in range(ty0, ty1 + 1):
            base = ty * ntx
            for tx in range(tx0, tx1 + 1):
                counts[base + tx] += 1
    starts = np.zeros(nt + 1, np.int64)
    for t in range(nt):
        starts[t + 1] = starts[t] + counts[t]
    total = starts[nt]
    ids = np.empty(total, np.int64)
    wp = starts[:nt].copy()
    for i in range(n):
        yy0 = tops[i]
        yy1 = tops[i] + hs[i] - 1
        xx0 = lefts[i]
        xx1 = lefts[i] + ws[i] - 1
        if yy0 < 0:
            yy0 = 0
        if xx0 < 0:
            xx0 = 0
        if yy1 > H - 1:
            yy1 = H - 1
        if xx1 > W - 1:
            xx1 = W - 1
        if yy0 > yy1 or xx0 > xx1:
            continue
        ty0 = yy0 // tile
        ty1 = yy1 // tile
        tx0 = xx0 // tile
        tx1 = xx1 // tile
        for ty in range(ty0, ty1 + 1):
            base = ty * ntx
            for tx in range(tx0, tx1 + 1):
                t = base + tx
                p = wp[t]
                ids[p] = i
                wp[t] = p + 1
    return starts, ids, ntx, nty


@njit(cache=True, nogil=True, parallel=True, fastmath=False)
def render_tiles(
    canvas: np.ndarray,
    tops: np.ndarray,
    lefts: np.ndarray,
    hs: np.ndarray,
    ws: np.ndarray,
    colors: np.ndarray,
    pool: np.ndarray,
    offsets: np.ndarray,
    starts: np.ndarray,
    ids: np.ndarray,
    ntx: int,
    nty: int,
    tile: int,
) -> None:
    H, W, C = canvas.shape
    nt = ntx * nty
    for t in prange(nt):
        ty = t // ntx
        tx = t - ty * ntx
        ty0 = ty * tile
        tx0 = tx * tile
        ty1 = ty0 + tile
        tx1 = tx0 + tile
        if ty1 > H:
            ty1 = H
        if tx1 > W:
            tx1 = W
        s0 = starts[t]
        s1 = starts[t + 1]
        for si in range(s0, s1):
            i = ids[si]
            top = tops[i]
            left = lefts[i]
            w = ws[i]
            off = offsets[i]
            oy0 = top if top > ty0 else ty0
            ox0 = left if left > tx0 else tx0
            oy1 = top + hs[i]
            ox1 = left + w
            if oy1 > ty1:
                oy1 = ty1
            if ox1 > tx1:
                ox1 = tx1
            if oy0 >= oy1 or ox0 >= ox1:
                continue
            cr = colors[i, 0]
            cg = colors[i, 1]
            cb = colors[i, 2]
            for y in range(oy0, oy1):
                row = off + (y - top) * w
                for x in range(ox0, ox1):
                    a = pool[row + (x - left)]
                    if a <= 0.0:
                        continue
                    ia = 1.0 - a
                    canvas[y, x, 0] = ia * canvas[y, x, 0] + a * cr
                    canvas[y, x, 1] = ia * canvas[y, x, 1] + a * cg
                    canvas[y, x, 2] = ia * canvas[y, x, 2] + a * cb


def render_tiled(
    canvas: np.ndarray,
    strokes: list[Stroke],
    tile: int = DEFAULT_TILE,
) -> np.ndarray:
    if not strokes:
        return canvas
    H, W = canvas.shape[:2]
    tops, lefts, hs, ws, colors, pool, offsets = pack_strokes(strokes)
    starts, ids, ntx, nty = bin_strokes_to_tiles(tops, lefts, hs, ws, H, W, tile)
    render_tiles(canvas, tops, lefts, hs, ws, colors, pool, offsets, starts, ids, ntx, nty, tile)
    return canvas


__all__ = [
    "Brush",
    "BrushShape",
    "DEFAULT_TILE",
    "RenderMode",
    "Stroke",
    "bin_strokes_to_tiles",
    "composite_over_bbox",
    "create_brush",
    "pack_strokes",
    "accumulate_coverage",
    "place",
    "rasterize_stroke",
    "render_coverage",
    "render_serial",
    "render_tiled",
    "render_tiles",
    "resample_path",
    "value_noise",
]
