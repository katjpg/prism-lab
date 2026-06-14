from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numba import njit

from prism.color.value import (
    DEFAULT_SEED,
    adjust_saturation,
    check_rgb,
    value_channel,
)
from prism.paint.ground import (
    blend_texture,
    create_canvas,
    reflect_border,
)
from prism.paint.region import render_underpaint
from prism.paint.style import (
    PaintConfig,
    PainterlyStyle,
    PaintResult,
    style_config,
)
from prism.paint.underpaint import Underpainting
from prism.preset import Detail, fit_image, preset, scaled_radii

EPS = 1e-8


@dataclass
class BrushStroke:
    path: tuple[tuple[float, float], ...]
    color: tuple[float, float, float]
    radius: float
    alpha: float
    angle: float


@dataclass(frozen=True, slots=True)
class StrokeCandidate:
    y: int
    x: int
    error: float


def paint(
    rgb: np.ndarray,
    style: PainterlyStyle = "impressionist",
    config: PaintConfig | None = None,
    detail: Detail = "standard",
    pixels: tuple[int, int] | None = None,
    pad: float = 0.06,
    underpainting: Underpainting | None = None,
    keep_overshoot: bool = False,
    seed: int = DEFAULT_SEED,
) -> PaintResult:
    check_rgb(rgb)

    cfg = style_config(style) if config is None else config
    rng = np.random.default_rng(seed)

    source, scale = fit_image(rgb, detail, pixels)
    source = adjust_saturation(source, cfg.saturation)

    background = cfg.background
    pad_px = int(round(pad * max(source.shape[:2]))) if pad > 0 else 0
    h, w = source.shape[:2]

    if underpainting is not None:
        canvas = render_underpaint(
            source,
            underpainting,
            background,
            seed,
            pad,
            (h + 2 * pad_px, w + 2 * pad_px),
        )
    else:
        canvas = create_canvas(source, background)
        if pad_px > 0:
            canvas = reflect_border(canvas, pad_px)

    if pad_px > 0:
        reference = reflect_border(create_canvas(source, background), pad_px)
        reference[pad_px : pad_px + h, pad_px : pad_px + w] = source
        source = reference

    use_kernel = True

    for i, radius in enumerate(scaled_radii(cfg.radii, detail)):
        ref = scale_space_image(source, radius, cfg.blur_factor)
        theta = tangent_field(ref)

        candidates = stroke_candidates(
            reference=ref,
            canvas=canvas,
            radius=radius,
            config=cfg,
            detail=detail,
            use_kernel=use_kernel,
            base_layer=(i == 0 and underpainting is None),
        )

        layer = [create_brush_stroke(c, ref, theta, radius, cfg, rng) for c in candidates]

        rng.shuffle(layer)

        for stroke in layer:
            paint_stroke(canvas, stroke)

    if pad_px > 0 and not keep_overshoot:
        canvas[:pad_px] = source[:pad_px]
        canvas[-pad_px:] = source[-pad_px:]
        canvas[:, :pad_px] = source[:, :pad_px]
        canvas[:, -pad_px:] = source[:, -pad_px:]

    if scale != 1.0:
        oh = round(canvas.shape[0] / scale)
        ow = round(canvas.shape[1] / scale)
        image = cv2.resize(
            canvas,
            (ow, oh),
            interpolation=cv2.INTER_LINEAR,
        ).astype("float32")
    else:
        image = canvas.astype("float32")

    image = np.clip(image, 0.0, 1.0).astype("float32")
    if background == "white":
        image = dehaze(image)
    image = blend_texture(image, background)

    return PaintResult(
        image=image,
        style=style,
        config=cfg,
        detail=detail,
    )


def dehaze(
    image: np.ndarray,
    white_point: float = 0.85,
    soft: float = 0.10,
) -> np.ndarray:
    mn = image.min(axis=2)
    t = np.clip((mn - white_point) / max(soft, EPS), 0.0, 1.0)[..., None]
    return (image * (1.0 - t) + t).astype("float32")


def scale_space_image(
    rgb: np.ndarray,
    radius: int,
    blur_factor: float,
) -> np.ndarray:
    sigma = max(0.1, blur_factor * radius)

    return cv2.GaussianBlur(
        rgb.astype("float32"),
        ksize=(0, 0),
        sigmaX=sigma,
    ).astype("float32")


def tangent_field(rgb: np.ndarray) -> np.ndarray:
    value = value_channel(rgb)
    gx = cv2.Sobel(value.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(value.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)

    return (np.arctan2(gy, gx) + np.pi / 2.0).astype("float32")


@njit(cache=True, nogil=True)
def candidate_kernel_fused(
    reference: np.ndarray,
    canvas: np.ndarray,
    step: int,
    threshold: float,
):
    h, w, _ = reference.shape
    ny = (h + step - 1) // step
    nx = (w + step - 1) // step
    max_n = ny * nx

    by = np.empty(max_n, dtype=np.int32)
    bx = np.empty(max_n, dtype=np.int32)
    errs = np.empty(max_n, dtype=np.float32)
    n = 0

    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        for x0 in range(0, w, step):
            x1 = min(w, x0 + step)

            total = 0.0
            count = 0
            best = np.float32(-1.0)
            cy = y0
            cx = x0

            for y in range(y0, y1):
                for x in range(x0, x1):
                    d0 = reference[y, x, 0] - canvas[y, x, 0]
                    d1 = reference[y, x, 1] - canvas[y, x, 1]
                    d2 = reference[y, x, 2] - canvas[y, x, 2]
                    e = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                    total += e
                    count += 1
                    if e > best:
                        best = e
                        cy = y
                        cx = x

            if count > 0 and total / count > threshold:
                by[n] = cy
                bx[n] = cx
                errs[n] = best
                n += 1

    return by[:n], bx[:n], errs[:n]


def stroke_candidates(
    reference: np.ndarray,
    canvas: np.ndarray,
    radius: int,
    config: PaintConfig,
    detail: Detail,
    use_kernel: bool = False,
    base_layer: bool = False,
) -> list[StrokeCandidate]:
    pset = preset(detail)
    step = max(1, round(config.grid_factor * radius * pset.grid))
    threshold = 0.0 if base_layer else float(config.threshold)

    if use_kernel:
        by, bx, errs = candidate_kernel_fused(
            np.ascontiguousarray(reference, dtype="float32"),
            np.ascontiguousarray(canvas, dtype="float32"),
            step,
            threshold,
        )
        return [StrokeCandidate(int(by[i]), int(bx[i]), float(errs[i])) for i in range(len(by))]

    error = residual_map(reference, canvas)
    h, w = error.shape
    candidates: list[StrokeCandidate] = []

    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)

        for x0 in range(0, w, step):
            x1 = min(w, x0 + step)
            cell = error[y0:y1, x0:x1]

            if cell.size == 0 or float(cell.mean()) <= threshold:
                continue

            yy, xx = np.unravel_index(int(np.argmax(cell)), cell.shape)
            y = int(y0 + yy)
            x = int(x0 + xx)

            candidates.append(StrokeCandidate(y=y, x=x, error=float(error[y, x])))

    return candidates


def create_brush_stroke(
    candidate: StrokeCandidate,
    reference: np.ndarray,
    theta: np.ndarray,
    radius: int,
    config: PaintConfig,
    rng: np.random.Generator,
) -> BrushStroke:
    x = candidate.x
    y = candidate.y

    color = stroke_color(reference[y, x], config.color_jitter, rng)
    angle = float(theta[y, x])

    if config.stroke_shape == "dot":
        path = ((float(x), float(y)),)
    else:
        path = trace_streamline(
            theta=theta,
            x=float(x),
            y=float(y),
            radius=float(radius),
            min_length=config.min_length,
            max_length=config.max_length,
            angle_jitter=config.angle_jitter,
            rng=rng,
        )

    return BrushStroke(
        path=path,
        color=(float(color[0]), float(color[1]), float(color[2])),
        radius=float(radius),
        alpha=float(config.alpha),
        angle=angle,
    )


def residual_map(
    reference: np.ndarray,
    canvas: np.ndarray,
) -> np.ndarray:
    diff = reference.astype("float32") - canvas.astype("float32")
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype("float32")


def stroke_color(
    color: np.ndarray,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(color, dtype="float32").copy()

    if jitter > 0:
        scale = 1.0 - jitter / 2.0 + jitter * rng.random(3)
        out *= scale.astype("float32")

    return np.clip(out, 0.0, 1.0).astype("float32")


def trace_streamline(
    theta: np.ndarray,
    x: float,
    y: float,
    radius: float,
    min_length: float,
    max_length: float,
    angle_jitter: float,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], ...]:
    if max_length <= 0:
        return ((x, y),)

    length = float(rng.uniform(min_length, max_length))
    step = max(1.0, radius * 0.75)
    n_steps = max(1, int(round(length / step)))

    backward = trace_direction(
        theta=theta,
        x=x,
        y=y,
        step=step,
        n_steps=n_steps // 2,
        direction=-1.0,
        angle_jitter=angle_jitter,
        rng=rng,
    )

    forward = trace_direction(
        theta=theta,
        x=x,
        y=y,
        step=step,
        n_steps=n_steps - n_steps // 2,
        direction=1.0,
        angle_jitter=angle_jitter,
        rng=rng,
    )

    pts = list(reversed(backward))
    pts.append((x, y))
    pts.extend(forward)

    return tuple((float(px), float(py)) for px, py in pts)


def trace_direction(
    theta: np.ndarray,
    x: float,
    y: float,
    step: float,
    n_steps: int,
    direction: float,
    angle_jitter: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    h, w = theta.shape
    pts: list[tuple[float, float]] = []

    px = x
    py = y
    prev_angle = sample_orientation(theta, px, py)

    for _ in range(n_steps):
        angle = sample_orientation(theta, px, py)

        if math.cos(angle - prev_angle) < 0:
            angle += math.pi

        if angle_jitter > 0:
            angle += float(rng.normal(0.0, angle_jitter))

        px += direction * step * math.cos(angle)
        py += direction * step * math.sin(angle)

        if px < 0 or px >= w or py < 0 or py >= h:
            break

        pts.append((px, py))
        prev_angle = angle

    return pts


def sample_orientation(
    theta: np.ndarray,
    x: float,
    y: float,
) -> float:
    h, w = theta.shape
    xx = min(max(int(round(x)), 0), w - 1)
    yy = min(max(int(round(y)), 0), h - 1)

    return float(theta[yy, xx])


def paint_stroke(
    canvas: np.ndarray,
    stroke: BrushStroke,
) -> None:
    path = np.asarray(stroke.path, dtype="float32")
    color = np.asarray(stroke.color, dtype="float32")

    if len(path) == 0:
        return

    paint_mask_stroke(canvas, path, color, stroke.radius, stroke.alpha)


def stroke_bbox(
    path: np.ndarray,
    radius: float,
    shape_hw: tuple[int, int],
    margin: float = 3.0,
) -> tuple[int, int, int, int]:
    h, w = shape_hw
    r = int(round(radius * margin))

    x0 = max(0, int(np.floor(float(path[:, 0].min()))) - r)
    x1 = min(w, int(np.ceil(float(path[:, 0].max()))) + r + 1)
    y0 = max(0, int(np.floor(float(path[:, 1].min()))) - r)
    y1 = min(h, int(np.ceil(float(path[:, 1].max()))) + r + 1)

    return x0, y0, x1, y1


def paint_mask_stroke(
    canvas: np.ndarray,
    path: np.ndarray,
    color: np.ndarray,
    radius: float,
    alpha: float,
) -> None:
    h, w = canvas.shape[:2]
    x0, y0, x1, y1 = stroke_bbox(path, radius, (h, w), margin=2.5)

    if x1 <= x0 or y1 <= y0:
        return

    local = np.zeros((y1 - y0, x1 - x0), dtype="float32")
    p = path.copy()
    p[:, 0] -= x0
    p[:, 1] -= y0

    r = max(1, int(round(radius * 0.8)))

    if len(p) == 1:
        x, y = p[0]
        cv2.circle(
            local,
            (int(round(float(x))), int(round(float(y)))),
            r,
            1.0,
            -1,
            lineType=cv2.LINE_AA,
        )
    else:
        pts = np.round(p).astype("int32").reshape(-1, 1, 2)
        cv2.polylines(
            local,
            [pts],
            isClosed=False,
            color=1.0,
            thickness=max(1, 2 * r),
            lineType=cv2.LINE_AA,
        )

    alpha_composite_inplace(canvas[y0:y1, x0:x1], local, color, alpha)


@njit(cache=True, nogil=True)
def alpha_composite_kernel(
    image: np.ndarray,
    alpha: np.ndarray,
    color: np.ndarray,
    opacity: float,
) -> None:
    h, w = alpha.shape
    op = np.float32(opacity)
    one = np.float32(1.0)
    zero = np.float32(0.0)
    c0 = color[0]
    c1 = color[1]
    c2 = color[2]

    for y in range(h):
        for x in range(w):
            a = alpha[y, x] * op
            if a <= zero:
                continue
            if a > one:
                a = one
            ia = one - a
            image[y, x, 0] = image[y, x, 0] * ia + c0 * a
            image[y, x, 1] = image[y, x, 1] * ia + c1 * a
            image[y, x, 2] = image[y, x, 2] * ia + c2 * a


def alpha_composite_inplace(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    alpha: float,
) -> None:
    alpha_composite_kernel(
        canvas,
        np.ascontiguousarray(mask, dtype="float32"),
        np.ascontiguousarray(color, dtype="float32"),
        float(alpha),
    )


__all__ = [
    "BrushStroke",
    "alpha_composite_inplace",
    "paint",
]
