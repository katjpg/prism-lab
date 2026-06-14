from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numba import njit
from scipy.spatial import cKDTree  # type: ignore

from prism.color.value import DEFAULT_SEED, value_channel
from prism.preset import Detail, fit_pixels, fit_side, preset

EPS = 1e-12

DEFAULT_N_POINTS = 8_000
DEFAULT_GAMMA = 1.4
DEFAULT_DOT_RADIUS = 0.75
DEFAULT_EDGE_WEIGHT = 0.35
DEFAULT_WHITE_CUTOFF = 0.08
DEFAULT_SMOOTH = 0.6

RELAX_STEPS: dict[Detail, int] = {"draft": 10, "standard": 18, "high": 26, "ultra": 40}
POINT_SCALE: dict[Detail, float] = {
    "draft": 0.7,
    "standard": 1.3,
    "high": 2.0,
    "ultra": 3.0,
}


@dataclass
class StippleMap:
    image: np.ndarray
    points: np.ndarray
    density: np.ndarray


def stipple(
    target: np.ndarray,
    n_points: int = DEFAULT_N_POINTS,
    detail: Detail = "standard",
    pixels: tuple[int, int] | None = None,
    gamma: float = DEFAULT_GAMMA,
    smooth: float = DEFAULT_SMOOTH,
    dot_radius: float = DEFAULT_DOT_RADIUS,
    edge_weight: float = DEFAULT_EDGE_WEIGHT,
    white_cutoff: float = DEFAULT_WHITE_CUTOFF,
    seed: int = DEFAULT_SEED,
    workers: int = -1,
) -> StippleMap:
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    if gamma <= 0:
        raise ValueError("gamma must be positive")

    pset = preset(detail)
    n_steps = RELAX_STEPS[detail]
    n_points = max(1, round(n_points * POINT_SCALE[detail]))

    if pixels is not None:
        target, _ = fit_pixels(target, pixels)

    value = image_value(target)

    h0, w0 = value.shape
    work_value, scale = fit_side(value, pset.side)

    if smooth > 0:
        work_value = cv2.GaussianBlur(
            work_value.astype("float32"),
            ksize=(0, 0),
            sigmaX=smooth,
        )

    density_work = density_from_value(
        work_value,
        gamma=gamma,
        edge_weight=edge_weight,
        white_cutoff=white_cutoff,
    )

    rng = np.random.default_rng(seed)
    coords = grid_coords(density_work.shape)
    weights = density_work.ravel().astype("float64")

    points = sample_points(
        coords=coords,
        weights=weights,
        n_points=n_points,
        rng=rng,
        shape_hw=density_work.shape,
    )

    points = relax_points(
        points=points,
        coords=coords,
        weights=weights,
        shape_hw=density_work.shape,
        n_steps=n_steps,
        rng=rng,
        workers=workers,
    )

    points_full = scale_points(points, scale=scale, shape_hw=(h0, w0))

    density_full = cv2.resize(
        density_work.astype("float32"),
        (w0, h0),
        interpolation=cv2.INTER_LINEAR,
    ).astype("float32")

    image = render_stipple(
        points_full,
        shape_hw=(h0, w0),
        radius=dot_radius,
    )

    return StippleMap(
        image=image,
        points=points_full.astype("float32"),
        density=density_full,
    )


def image_value(target: np.ndarray) -> np.ndarray:
    if target.ndim == 2:
        value = target.astype("float32")
    elif target.ndim == 3 and target.shape[-1] == 3:
        value = value_channel(target.astype("float32"))
    else:
        raise ValueError(f"expected image with shape (H, W) or (H, W, 3), got {target.shape}")

    if not np.isfinite(value).all():
        raise ValueError("target contains non-finite values")

    return np.clip(value, 0.0, 1.0).astype("float32")


def density_from_value(
    value: np.ndarray,
    gamma: float = DEFAULT_GAMMA,
    edge_weight: float = DEFAULT_EDGE_WEIGHT,
    white_cutoff: float = DEFAULT_WHITE_CUTOFF,
    black_point: float = 2.0,
    white_point: float = 98.0,
) -> np.ndarray:
    lo, hi = np.percentile(value, (black_point, white_point))
    v = (value - lo) / max(float(hi - lo), EPS)
    v = np.clip(v, 0.0, 1.0)

    tone = np.power(1.0 - v, gamma)
    tone = np.clip(tone - white_cutoff, 0.0, 1.0)

    gx = cv2.Sobel(v.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(v.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(gx, gy)

    if float(edge.max()) > EPS:
        edge = edge / float(edge.max())

    density = np.clip(tone + edge_weight * edge, 0.0, 1.0)

    if float(density.sum()) <= EPS:
        return np.ones(value.shape, dtype="float32")

    return density.astype("float32")


def grid_coords(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w]

    return np.column_stack(
        [xx.ravel(), yy.ravel()],
    ).astype("float64")


def sample_points(
    coords: np.ndarray,
    weights: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    valid = weights > EPS
    if not np.any(valid):
        valid = np.ones(len(weights), dtype=bool)

    coords_valid = coords[valid]
    weights_valid = weights[valid]
    prob = weights_valid / weights_valid.sum()

    replace = n_points > len(coords_valid)
    idx = rng.choice(
        len(coords_valid),
        size=n_points,
        replace=replace,
        p=prob,
    )

    points = coords_valid[idx].astype("float64")
    points += rng.uniform(-0.5, 0.5, size=points.shape)

    h, w = shape_hw
    points[:, 0] = np.clip(points[:, 0], 0.0, w - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, h - 1.0)

    return points


def relax_points(
    points: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray,
    shape_hw: tuple[int, int],
    n_steps: int,
    rng: np.random.Generator,
    workers: int = -1,
) -> np.ndarray:
    out = points.astype("float64", copy=True)

    for _ in range(n_steps):
        labels = nearest_points(
            coords=coords,
            points=out,
            workers=workers,
        )

        out, mass = weighted_centroids(
            points=out,
            labels=labels,
            coords=coords,
            weights=weights,
            shape_hw=shape_hw,
        )

        dead = mass <= EPS
        if np.any(dead):
            out[dead] = sample_points(
                coords=coords,
                weights=weights,
                n_points=int(dead.sum()),
                rng=rng,
                shape_hw=shape_hw,
            )

    return out


def nearest_points(
    coords: np.ndarray,
    points: np.ndarray,
    workers: int = -1,
) -> np.ndarray:
    tree = cKDTree(points)

    try:
        _, labels = tree.query(coords, k=1, workers=workers)
    except TypeError:
        _, labels = tree.query(coords, k=1)

    return labels.astype("int64")


def weighted_centroids(
    points: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    out, mass = _weighted_centroids(
        points=points,
        labels=labels,
        coords=coords,
        weights=weights,
    )

    h, w = shape_hw
    out[:, 0] = np.clip(out[:, 0], 0.0, w - 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, h - 1.0)

    return out, mass


@njit(cache=True, nogil=True)
def _weighted_centroids(points, labels, coords, weights):
    n_points = points.shape[0]

    mass = np.zeros(n_points, dtype=np.float64)
    sum_x = np.zeros(n_points, dtype=np.float64)
    sum_y = np.zeros(n_points, dtype=np.float64)

    for i in range(labels.shape[0]):
        k = labels[i]
        w = weights[i]

        mass[k] += w
        sum_x[k] += coords[i, 0] * w
        sum_y[k] += coords[i, 1] * w

    out = points.copy()

    for k in range(n_points):
        if mass[k] > EPS:
            out[k, 0] = sum_x[k] / mass[k]
            out[k, 1] = sum_y[k] / mass[k]

    return out, mass


def scale_points(
    points: np.ndarray,
    scale: float,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    h, w = shape_hw

    out = points.astype("float64", copy=True)
    out *= scale

    out[:, 0] = np.clip(out[:, 0], 0.0, w - 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, h - 1.0)

    return out


def render_stipple(
    points: np.ndarray,
    shape_hw: tuple[int, int],
    radius: float = DEFAULT_DOT_RADIUS,
) -> np.ndarray:
    h, w = shape_hw
    canvas = np.full((h, w), 255, dtype="uint8")

    r = max(1, int(round(radius)))

    for x, y in points:
        cx = int(round(float(x)))
        cy = int(round(float(y)))

        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue

        cv2.circle(
            canvas,
            center=(cx, cy),
            radius=r,
            color=0,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    return (canvas.astype("float32") / 255.0).astype("float32")


__all__ = [
    "StippleMap",
    "density_from_value",
    "grid_coords",
    "image_value",
    "nearest_points",
    "relax_points",
    "render_stipple",
    "sample_points",
    "scale_points",
    "stipple",
    "weighted_centroids",
]
