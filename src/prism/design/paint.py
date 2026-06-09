from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from prism.color.value import DEFAULT_SEED, value_channel
from prism.design.style import PaintConfig, PainterlyStyle, style_config


DEFAULT_WORK_MAX_SIDE = 700
EPS = 1e-8


@dataclass
class BrushStroke:
    path: tuple[tuple[float, float], ...]
    color: tuple[float, float, float]
    radius: float
    alpha: float
    angle: float


@dataclass
class PaintResult:
    image: np.ndarray
    strokes: list[BrushStroke]
    style: PainterlyStyle
    config: PaintConfig


def paint(
    rgb: np.ndarray,
    style: PainterlyStyle = "impressionist",
    config: PaintConfig | None = None,
    work_size: int = DEFAULT_WORK_MAX_SIDE,
    seed: int = DEFAULT_SEED,
) -> PaintResult:
    check_rgb(rgb)

    cfg = style_config(style) if config is None else config
    rng = np.random.default_rng(seed)

    source, scale = resize_max_side(rgb, work_size)
    source = adjust_saturation(source, cfg.saturation)

    canvas = initial_canvas(source, cfg.background)
    strokes: list[BrushStroke] = []

    for radius in cfg.radii:
        ref = blurred_reference(source, radius, cfg.blur_factor)
        theta = orientation_map(ref)

        layer = make_layer_strokes(
            reference=ref,
            canvas=canvas,
            theta=theta,
            radius=radius,
            config=cfg,
            rng=rng,
        )

        rng.shuffle(layer)

        for stroke in layer:
            apply_stroke(canvas, stroke)
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
        style=style,
        config=cfg,
    )


def check_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected RGB image with shape (H, W, 3), got {rgb.shape}")

    if not np.isfinite(rgb).all():
        raise ValueError("RGB image contains non-finite values")

    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise ValueError("RGB image must be normalized to [0, 1]")


def resize_max_side(
    rgb: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]

    if max_side <= 0 or max(h, w) <= max_side:
        return rgb.astype("float32"), 1.0

    scale_down = max_side / max(h, w)
    hh = max(1, round(h * scale_down))
    ww = max(1, round(w * scale_down))

    out = cv2.resize(
        rgb.astype("float32"),
        (ww, hh),
        interpolation=cv2.INTER_AREA,
    )

    scale_up = max(h, w) / max(hh, ww)
    return out.astype("float32"), float(scale_up)


def initial_canvas(
    rgb: np.ndarray,
    background: str,
) -> np.ndarray:
    h, w = rgb.shape[:2]

    if background == "source":
        return rgb.astype("float32").copy()

    if background == "white":
        return np.ones((h, w, 3), dtype="float32")

    if background == "black":
        return np.zeros((h, w, 3), dtype="float32")

    if background == "mean":
        color = rgb.mean(axis=(0, 1))
        canvas = np.empty((h, w, 3), dtype="float32")
        canvas[:] = color
        return canvas

    raise ValueError(f"unknown background: {background!r}")


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


def blurred_reference(
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


def orientation_map(rgb: np.ndarray) -> np.ndarray:
    value = value_channel(rgb)
    gx = cv2.Sobel(value.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(value.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)

    return (np.arctan2(gy, gx) + np.pi / 2.0).astype("float32")


def make_layer_strokes(
    reference: np.ndarray,
    canvas: np.ndarray,
    theta: np.ndarray,
    radius: int,
    config: PaintConfig,
    rng: np.random.Generator,
) -> list[BrushStroke]:
    error = color_error(reference, canvas)
    h, w = error.shape
    step = max(1, round(config.grid_factor * radius))

    strokes: list[BrushStroke] = []

    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)

        for x0 in range(0, w, step):
            x1 = min(w, x0 + step)
            cell = error[y0:y1, x0:x1]

            if cell.size == 0 or float(cell.mean()) <= config.threshold:
                continue

            yy, xx = np.unravel_index(int(np.argmax(cell)), cell.shape)
            y = y0 + yy
            x = x0 + xx

            color = stroke_color(reference[y, x], config.color_jitter, rng)
            angle = float(theta[y, x])

            if config.stroke_shape == "dot":
                path = ((float(x), float(y)),)
            else:
                path = curved_path(
                    theta=theta,
                    x=float(x),
                    y=float(y),
                    radius=float(radius),
                    min_length=config.min_length,
                    max_length=config.max_length,
                    angle_jitter=config.angle_jitter,
                    rng=rng,
                )

            strokes.append(
                BrushStroke(
                    path=path,
                    color=(float(color[0]), float(color[1]), float(color[2])),
                    radius=float(radius),
                    alpha=float(config.alpha),
                    angle=angle,
                )
            )

    return strokes


def color_error(
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


def curved_path(
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

    backward = trace_path(
        theta=theta,
        x=x,
        y=y,
        step=step,
        n_steps=n_steps // 2,
        direction=-1.0,
        angle_jitter=angle_jitter,
        rng=rng,
    )

    forward = trace_path(
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


def trace_path(
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
    prev_angle = sample_angle(theta, px, py)

    for _ in range(n_steps):
        angle = sample_angle(theta, px, py)

        if np.cos(angle - prev_angle) < 0:
            angle += np.pi

        if angle_jitter > 0:
            angle += float(rng.normal(0.0, angle_jitter))

        px += direction * step * float(np.cos(angle))
        py += direction * step * float(np.sin(angle))

        if px < 0 or px >= w or py < 0 or py >= h:
            break

        pts.append((px, py))
        prev_angle = angle

    return pts


def sample_angle(
    theta: np.ndarray,
    x: float,
    y: float,
) -> float:
    h, w = theta.shape
    xx = int(np.clip(round(x), 0, w - 1))
    yy = int(np.clip(round(y), 0, h - 1))

    return float(theta[yy, xx])


def apply_stroke(
    canvas: np.ndarray,
    stroke: BrushStroke,
) -> None:
    if len(stroke.path) == 1:
        apply_dot(canvas, stroke)
    else:
        apply_polyline(canvas, stroke)


def apply_dot(
    canvas: np.ndarray,
    stroke: BrushStroke,
) -> None:
    h, w = canvas.shape[:2]
    x, y = stroke.path[0]
    cx, cy = int(round(x)), int(round(y))
    r = max(1, int(round(stroke.radius)))
    pad = r + 2

    x0 = max(0, cx - pad)
    x1 = min(w, cx + pad + 1)
    y0 = max(0, cy - pad)
    y1 = min(h, cy + pad + 1)

    if x1 <= x0 or y1 <= y0:
        return

    mask = np.zeros((y1 - y0, x1 - x0), dtype="uint8")
    cv2.circle(
        mask,
        center=(cx - x0, cy - y0),
        radius=r,
        color=255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    composite(canvas, mask, stroke.color, stroke.alpha, y0, x0)


def apply_polyline(
    canvas: np.ndarray,
    stroke: BrushStroke,
) -> None:
    h, w = canvas.shape[:2]
    pts = np.asarray(stroke.path, dtype="float32")

    if len(pts) < 2:
        apply_dot(canvas, stroke)
        return

    thickness = max(1, int(round(2.0 * stroke.radius)))
    pad = thickness // 2 + 2

    x0 = max(0, int(np.floor(float(pts[:, 0].min()))) - pad)
    x1 = min(w, int(np.ceil(float(pts[:, 0].max()))) + pad + 1)
    y0 = max(0, int(np.floor(float(pts[:, 1].min()))) - pad)
    y1 = min(h, int(np.ceil(float(pts[:, 1].max()))) + pad + 1)

    if x1 <= x0 or y1 <= y0:
        return

    mask = np.zeros((y1 - y0, x1 - x0), dtype="uint8")
    local = pts.round().astype("int32") - np.array([x0, y0], dtype="int32")
    cv2.polylines(
        mask,
        [local],
        isClosed=False,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    composite(canvas, mask, stroke.color, stroke.alpha, y0, x0)


def composite(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
    y0: int,
    x0: int,
) -> None:
    a = (mask.astype("float32") / 255.0) * alpha

    if float(a.max()) <= 0:
        return

    c = np.asarray(color, dtype="float32")
    mh, mw = mask.shape
    region = canvas[y0 : y0 + mh, x0 : x0 + mw]
    region[:] = region * (1.0 - a[..., None]) + c * a[..., None]


def scale_strokes(
    strokes: list[BrushStroke],
    scale: float,
) -> list[BrushStroke]:
    out: list[BrushStroke] = []

    for stroke in strokes:
        path = tuple((x * scale, y * scale) for x, y in stroke.path)

        out.append(
            BrushStroke(
                path=path,
                color=stroke.color,
                radius=stroke.radius * scale,
                alpha=stroke.alpha,
                angle=stroke.angle,
            )
        )

    return out


__all__ = [
    "BrushStroke",
    "PaintResult",
    "adjust_saturation",
    "blurred_reference",
    "color_error",
    "curved_path",
    "initial_canvas",
    "orientation_map",
    "paint",
    "resize_max_side",
    "stroke_color",
]
