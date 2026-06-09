from __future__ import annotations

import types

import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image, skeletonize

from prism.color.value import DEFAULT_SEED
from prism.design.brush import Brush, RenderMode
from prism.design.paint import (
    BrushStroke,
    PaintResult,
    adjust_saturation,
    apply_bristle_stroke,
    check_rgb,
    initial_canvas,
    resize_max_side,
    scale_strokes,
    stroke_color,
)
from prism.form.region import superpixels


DEFAULT_WORK_MAX_SIDE = 700
DEFAULT_PASSES = (
    (120, 12.0),
    (300, 9.0),
    (700, 6.5),
    (1400, 4.5),
    (2600, 3.2),
    (4200, 2.4),
)
EPS = 1e-8


def skeleton_path(mask: np.ndarray, n_points: int = 5) -> np.ndarray:
    hull = convex_hull_image(mask)
    skel = skeletonize(hull)

    pts = np.argwhere(skel)
    if len(pts) == 0:
        pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype="float32")

    xy = pts[:, ::-1].astype("float32")
    if len(xy) <= 2:
        return xy

    centered = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[0]

    order = np.argsort(proj)
    xy = xy[order]
    proj = proj[order]

    targets = np.linspace(proj[0], proj[-1], n_points)
    idx = np.unique(np.clip(np.searchsorted(proj, targets), 0, len(xy) - 1))

    return xy[idx]


def make_segment_strokes(
    source: np.ndarray,
    labels: np.ndarray,
    radius: float,
    color_jitter: float,
    alpha: float,
    overlap: float,
    rng: np.random.Generator,
) -> list[BrushStroke]:
    strokes: list[BrushStroke] = []

    for region in regionprops(labels):
        minr, minc, _, _ = region.bbox
        path = skeleton_path(region.image)

        if len(path) >= 2:
            path = path + np.array([minc, minr], dtype="float32")
        else:
            cy, cx = region.centroid
            angle = float(region.orientation)
            length = 0.5 * max(float(region.axis_major_length), 2.0)
            dx = np.cos(angle) * length
            dy = -np.sin(angle) * length
            path = np.array(
                [[cx - dx, cy - dy], [cx + dx, cy + dy]],
                dtype="float32",
            )

        coords = region.coords
        mean = source[coords[:, 0], coords[:, 1]].mean(axis=0)
        color = stroke_color(mean, color_jitter, rng)

        rad = max(radius, 0.5 * float(region.axis_minor_length)) * overlap
        rad = max(rad, 1.0)

        strokes.append(
            BrushStroke(
                path=tuple((float(x), float(y)) for x, y in path),
                color=(float(color[0]), float(color[1]), float(color[2])),
                radius=float(rad),
                alpha=float(alpha),
                angle=float(region.orientation),
            )
        )

    return strokes


def paint_segments(
    rgb: np.ndarray,
    passes: tuple[tuple[int, float], ...] = DEFAULT_PASSES,
    detail: float = 1.0,
    compactness: float = 3.0,
    saturation: float = 1.1,
    background: str = "white",
    pad: float = 0.06,
    alpha: float = 0.90,
    color_jitter: float = 0.08,
    overlap: float = 1.4,
    vary: int = 24,
    bristle_count: int = 48,
    brush_shape: str = "filbert",
    brush_drag: float = 0.08,
    brush_step: float = 1.5,
    brush_samples: int = 12,
    dryness: float = 0.0,
    grain: float = 0.0,
    rotate_brush: bool = True,
    bristle_render: RenderMode = "line",
    work_size: int = DEFAULT_WORK_MAX_SIDE,
    seed: int = DEFAULT_SEED,
) -> PaintResult:
    check_rgb(rgb)

    if pad > 0:
        p = int(round(pad * max(rgb.shape[:2])))
        if p > 0:
            rgb = np.pad(rgb, ((p, p), (p, p), (0, 0)), constant_values=1.0)

    rng = np.random.default_rng(seed)

    source, scale = resize_max_side(rgb, work_size)
    source = adjust_saturation(source, saturation)

    canvas = initial_canvas(source, background)

    config = types.SimpleNamespace(
        bristle_vary=vary,
        bristle_count=bristle_count,
        brush_shape=brush_shape,
        brush_drag=brush_drag,
        brush_step=brush_step,
        brush_samples=brush_samples,
        dryness=dryness,
        grain=grain,
        rotate_brush=rotate_brush,
        bristle_render=bristle_render,
    )

    brush_cache: dict[object, Brush] = {}
    strokes: list[BrushStroke] = []

    for n_segments, radius in passes:
        n = max(1, int(round(n_segments * detail)))
        labels = superpixels(source, n_segments=n, compactness=compactness)

        layer = make_segment_strokes(
            source=source,
            labels=labels,
            radius=float(radius),
            color_jitter=color_jitter,
            alpha=alpha,
            overlap=overlap,
            rng=rng,
        )

        rng.shuffle(layer)

        for j, stroke in enumerate(layer):
            apply_bristle_stroke(
                canvas=canvas,
                stroke=stroke,
                config=config,  # type: ignore[arg-type]
                brush_cache=brush_cache,
                seed=seed + 7919 * int(round(radius)) + j,
            )
            strokes.append(stroke)

    h0, w0 = rgb.shape[:2]

    if scale != 1.0:
        image = cv2.resize(
            canvas,
            (w0, h0),
            interpolation=cv2.INTER_LINEAR,
        ).astype("float32")
        out_strokes = scale_strokes(strokes, scale)
    else:
        image = canvas.astype("float32")
        out_strokes = strokes

    return PaintResult(
        image=np.clip(image, 0.0, 1.0).astype("float32"),
        strokes=out_strokes,
        style="acrylic",
        config=config,  # type: ignore[arg-type]
    )


__all__ = [
    "make_segment_strokes",
    "paint_segments",
    "skeleton_path",
]
