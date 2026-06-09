from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit


BrushShape = Literal["round", "flat", "filbert"]
RenderMode = Literal["splat", "line"]


EPS = 1e-8

DEFAULT_SEED = 7
DEFAULT_STEP = 1.5
DEFAULT_SAMPLES = 12

DEFAULT_RADIUS = 12.0
DEFAULT_N_BRISTLES = 48
DEFAULT_DRAG = 0.08
DEFAULT_ALPHA = 0.90
DEFAULT_DRYNESS = 0.18
DEFAULT_GRAIN = 0.12


@dataclass
class Brush:
    xy: np.ndarray
    L: np.ndarray
    rho: np.ndarray
    r: np.ndarray
    a: np.ndarray
    eta: np.ndarray
    shape: str


@dataclass
class Stroke:
    mask: np.ndarray
    rgb: np.ndarray


def make_brush(
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

    xy = _sample_offsets(
        radius=radius,
        n=n_bristles,
        shape=shape,
        rng=rng,
    )

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


def apply_brushstroke(
    canvas: np.ndarray,
    path: np.ndarray,
    color: np.ndarray,
    brush: Brush,
    alpha: float = DEFAULT_ALPHA,
    step: float = DEFAULT_STEP,
    samples: int = DEFAULT_SAMPLES,
    drag: float = DEFAULT_DRAG,
    dryness: float = DEFAULT_DRYNESS,
    grain: float = DEFAULT_GRAIN,
    rotate: bool = False,
    render: RenderMode = "splat",
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, Stroke]:
    base = _as_rgb01(canvas)
    h, w, _ = base.shape

    color = np.asarray(color, dtype=np.float32)
    if color.ndim != 1 or color.shape[0] != 3:
        raise ValueError("color must have shape (3,)")

    if color.max() > 1.0:
        color = color / 255.0
    color = np.clip(color, 0.0, 1.0).astype(np.float32)

    p = resample_path(path, step=step)
    if len(p) < 2:
        return base.copy(), Stroke(
            mask=np.zeros((h, w), dtype=np.float32), rgb=base.copy()
        )

    mask = stroke_mask(
        shape_hw=(h, w),
        path=p,
        brush=brush,
        samples=samples,
        drag=drag,
        dryness=dryness,
        grain=grain,
        rotate=rotate,
        render=render,
        seed=seed,
    )

    out = composite(base, mask, color, alpha=alpha)
    return out, Stroke(mask=mask, rgb=out)


def stroke_mask(
    shape_hw: tuple[int, int],
    path: np.ndarray,
    brush: Brush,
    samples: int = DEFAULT_SAMPLES,
    drag: float = DEFAULT_DRAG,
    dryness: float = DEFAULT_DRYNESS,
    grain: float = DEFAULT_GRAIN,
    rotate: bool = False,
    render: RenderMode = "splat",
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    h, w = shape_hw
    p = np.asarray(path, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("path must have shape (N, 2)")
    if len(p) < 2:
        return np.zeros((h, w), dtype=np.float32)

    rng = np.random.default_rng(seed)

    d0 = p[1] - p[0]
    if _norm(d0) < EPS:
        d0 = np.array([1.0, 0.0], dtype=np.float32)
    d0 = d0 / max(_norm(d0), EPS)

    n = brush.xy.shape[0]

    p2 = (-d0[None, :] * brush.L[:, None]).astype(np.float32)
    p1 = (0.5 * p2).astype(np.float32)

    accum = np.zeros((h, w), dtype=np.float32)

    for k in range(1, len(p)):
        d = p[k] - p[k - 1]
        if _norm(d) < EPS:
            continue

        noise = rng.uniform(0.0, 0.25, size=n).astype(np.float32)

        p1, p2 = _advance_bristles(
            p1=p1,
            p2=p2,
            d=d.astype(np.float32),
            L=brush.L,
            rho=brush.rho,
            eta=brush.eta,
            noise=noise,
            drag=np.float32(drag),
        )

        if rotate:
            theta = np.arctan2(float(d[1]), float(d[0]))
            xy = _rotate_offsets(brush.xy, theta)
        else:
            xy = brush.xy

        if render == "line":
            _render_step_line(
                accum=accum,
                cx=float(p[k, 0]),
                cy=float(p[k, 1]),
                xy=xy.astype(np.float32),
                p1=p1,
                p2=p2,
                aa=brush.a,
                samples=samples,
            )
        else:
            _render_step(
                accum=accum,
                cx=float(p[k, 0]),
                cy=float(p[k, 1]),
                xy=xy.astype(np.float32),
                p1=p1,
                p2=p2,
                rr=brush.r,
                aa=brush.a,
                samples=samples,
            )

    if dryness > 0 or grain > 0:
        tex = texture_map(
            shape_hw=(h, w),
            seed=seed + 101,
            sigma=max(0.0, 0.75 + 5.0 * grain),
        )
        mask = accum * (1.0 - dryness + dryness * tex)
    else:
        mask = accum

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def composite(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    base = _as_rgb01(canvas)
    m = np.clip(mask.astype(np.float32) * float(alpha), 0.0, 1.0)[..., None]
    c = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(base * (1.0 - m) + c * m, 0.0, 1.0).astype(np.float32)


def stroke_layer(
    shape_hw: tuple[int, int],
    path: np.ndarray,
    color: np.ndarray,
    brush: Brush,
    alpha: float = DEFAULT_ALPHA,
    step: float = DEFAULT_STEP,
    samples: int = DEFAULT_SAMPLES,
    drag: float = DEFAULT_DRAG,
    dryness: float = DEFAULT_DRYNESS,
    grain: float = DEFAULT_GRAIN,
    rotate: bool = False,
    render: RenderMode = "splat",
    seed: int = DEFAULT_SEED,
) -> Stroke:
    h, w = shape_hw
    p = resample_path(path, step=step)
    mask = stroke_mask(
        shape_hw=(h, w),
        path=p,
        brush=brush,
        samples=samples,
        drag=drag,
        dryness=dryness,
        grain=grain,
        rotate=rotate,
        render=render,
        seed=seed,
    )
    rgb = np.ones((h, w, 3), dtype=np.float32)
    rgb = composite(rgb, mask, color, alpha=alpha)
    return Stroke(mask=mask, rgb=rgb)


def resample_path(path: np.ndarray, step: float = DEFAULT_STEP) -> np.ndarray:
    p = np.asarray(path, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("path must have shape (N, 2)")
    if len(p) == 0:
        return p.copy()
    if len(p) == 1 or step <= 0:
        return p.copy()

    out = [p[0]]
    for i in range(1, len(p)):
        a = p[i - 1]
        b = p[i]
        d = b - a
        L = float(np.hypot(d[0], d[1]))
        if L < EPS:
            continue
        m = max(1, int(np.ceil(L / step)))
        for j in range(1, m + 1):
            t = j / m
            out.append((1.0 - t) * a + t * b)

    return np.asarray(out, dtype=np.float32)


def texture_map(
    shape_hw: tuple[int, int],
    seed: int = DEFAULT_SEED,
    sigma: float = 1.5,
) -> np.ndarray:
    h, w = shape_hw
    rng = np.random.default_rng(seed)
    x = rng.random((h, w), dtype=np.float32)

    if sigma <= 0:
        return x

    k = max(1, int(round(3.0 * sigma)))
    y = _box_blur_sep(x, k)
    y = _box_blur_sep(y, k)
    y = _box_blur_sep(y, k)

    y = y - y.min()
    den = max(float(y.max()), EPS)
    return (y / den).astype(np.float32)


def _sample_offsets(
    radius: float,
    n: int,
    shape: BrushShape,
    rng: np.random.Generator,
) -> np.ndarray:
    if shape == "round":
        t = rng.uniform(0.0, 2.0 * np.pi, size=n)
        r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y])

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

    raise ValueError(f"unknown brush shape: {shape}")


def _rotate_offsets(xy: np.ndarray, theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return xy @ R.T


def _as_rgb01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"expected image with shape (H, W, 3), got {x.shape}")
    y = x.astype(np.float32)
    if y.max() > 1.0:
        y = y / 255.0
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _norm(v: np.ndarray) -> float:
    return float(np.hypot(float(v[0]), float(v[1])))


def _box_blur_sep(x: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return x.astype(np.float32, copy=True)

    y = np.empty_like(x, dtype=np.float32)
    z = np.empty_like(x, dtype=np.float32)

    pad = r

    xp = np.pad(x.astype(np.float32), ((0, 0), (pad, pad)), mode="reflect")
    cs = np.cumsum(xp, axis=1, dtype=np.float32)
    y[:] = (cs[:, 2 * pad :] - cs[:, : -2 * pad]) / max(2 * pad, 1)

    yp = np.pad(y.astype(np.float32), ((pad, pad), (0, 0)), mode="reflect")
    cs = np.cumsum(yp, axis=0, dtype=np.float32)
    z[:] = (cs[2 * pad :, :] - cs[: -2 * pad, :]) / max(2 * pad, 1)

    return z.astype(np.float32)


@njit(cache=True, fastmath=True)
def _unit_xy(x: float, y: float) -> tuple[float, float]:
    n = np.sqrt(x * x + y * y)
    if n < EPS:
        return 0.0, 0.0
    return x / n, y / n


@njit(cache=True, fastmath=True)
def _advance_bristles(
    p1: np.ndarray,
    p2: np.ndarray,
    d: np.ndarray,
    L: np.ndarray,
    rho: np.ndarray,
    eta: np.ndarray,
    noise: np.ndarray,
    drag: np.float32,
):
    n = p1.shape[0]
    q1 = np.empty_like(p1)
    q2 = np.empty_like(p2)

    dx = -float(d[0])
    dy = -float(d[1])
    tx, ty = _unit_xy(dx, dy)

    for i in range(n):
        ux, uy = _unit_xy(float(p2[i, 0]), float(p2[i, 1]))

        mx = float(eta[i]) * ux + (1.0 - float(eta[i])) * tx
        my = float(eta[i]) * uy + (1.0 - float(eta[i])) * ty
        mx, my = _unit_xy(mx, my)

        t2x = float(L[i]) * mx
        t2y = float(L[i]) * my

        nx = float(p2[i, 0]) + float(rho[i]) * (t2x - float(p2[i, 0]))
        ny = float(p2[i, 1]) + float(rho[i]) * (t2y - float(p2[i, 1]))
        nx, ny = _unit_xy(nx, ny)
        nx *= float(L[i])
        ny *= float(L[i])

        cx = 0.5 * nx
        cy = 0.5 * ny

        a = float(noise[i])
        cx = (1.0 - a) * cx + a * float(p1[i, 0])
        cy = (1.0 - a) * cy + a * float(p1[i, 1])

        q1[i, 0] = np.float32((1.0 - drag) * cx + drag * 0.5 * float(p1[i, 0]))
        q1[i, 1] = np.float32((1.0 - drag) * cy + drag * 0.5 * float(p1[i, 1]))
        q2[i, 0] = np.float32(nx)
        q2[i, 1] = np.float32(ny)

    return q1, q2


@njit(cache=True, fastmath=True)
def _render_step(
    accum: np.ndarray,
    cx: float,
    cy: float,
    xy: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    rr: np.ndarray,
    aa: np.ndarray,
    samples: int,
) -> None:
    n = xy.shape[0]
    for i in range(n):
        x0 = cx + float(xy[i, 0])
        y0 = cy + float(xy[i, 1])
        x1 = x0 + float(p1[i, 0])
        y1 = y0 + float(p1[i, 1])
        x2 = x0 + float(p2[i, 0])
        y2 = y0 + float(p2[i, 1])

        _render_quad(
            accum=accum,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            radius=float(rr[i]),
            alpha=float(aa[i]),
            samples=samples,
        )


@njit(cache=True, fastmath=True)
def _render_quad(
    accum: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    alpha: float,
    samples: int,
) -> None:
    if samples < 2:
        samples = 2

    for k in range(samples + 1):
        t = k / samples
        s = 1.0 - t
        x = s * s * x0 + 2.0 * s * t * x1 + t * t * x2
        y = s * s * y0 + 2.0 * s * t * y1 + t * t * y2

        taper = 1.0 - 0.25 * t
        _splat(
            accum=accum,
            x=x,
            y=y,
            radius=radius,
            alpha=alpha * taper,
        )


@njit(cache=True, fastmath=True)
def _splat(
    accum: np.ndarray,
    x: float,
    y: float,
    radius: float,
    alpha: float,
) -> None:
    h, w = accum.shape
    r = max(radius, 0.5)
    r2 = r * r

    x0 = max(0, int(np.floor(x - r)))
    x1 = min(w - 1, int(np.ceil(x + r)))
    y0 = max(0, int(np.floor(y - r)))
    y1 = min(h - 1, int(np.ceil(y + r)))

    for yy in range(y0, y1 + 1):
        dy = (yy + 0.5) - y
        for xx in range(x0, x1 + 1):
            dx = (xx + 0.5) - x
            d2 = dx * dx + dy * dy
            if d2 > r2:
                continue

            wgt = 1.0 - d2 / (r2 + EPS)
            val = alpha * wgt
            nv = accum[yy, xx] + val
            accum[yy, xx] = 1.0 if nv > 1.0 else nv


@njit(cache=True, fastmath=True)
def _render_step_line(
    accum: np.ndarray,
    cx: float,
    cy: float,
    xy: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    aa: np.ndarray,
    samples: int,
) -> None:
    n = xy.shape[0]
    for i in range(n):
        x0 = cx + float(xy[i, 0])
        y0 = cy + float(xy[i, 1])
        x1 = x0 + float(p1[i, 0])
        y1 = y0 + float(p1[i, 1])
        x2 = x0 + float(p2[i, 0])
        y2 = y0 + float(p2[i, 1])

        _render_quad_line(
            accum=accum,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            alpha=float(aa[i]),
            samples=samples,
        )


@njit(cache=True, fastmath=True)
def _render_quad_line(
    accum: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    alpha: float,
    samples: int,
) -> None:
    if samples < 2:
        samples = 2

    px = x0
    py = y0
    for k in range(1, samples + 1):
        t = k / samples
        s = 1.0 - t
        x = s * s * x0 + 2.0 * s * t * x1 + t * t * x2
        y = s * s * y0 + 2.0 * s * t * y1 + t * t * y2

        taper = 1.0 - 0.25 * t
        _line_accum(accum, px, py, x, y, alpha * taper)

        px = x
        py = y


@njit(cache=True, fastmath=True)
def _line_accum(
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
            nv = accum[y, x] + alpha
            accum[y, x] = 1.0 if nv > 1.0 else nv

        if x == ix1 and y == iy1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


__all__ = [
    "Brush",
    "BrushShape",
    "RenderMode",
    "Stroke",
    "apply_brushstroke",
    "composite",
    "make_brush",
    "resample_path",
    "stroke_layer",
    "stroke_mask",
    "texture_map",
]
