from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from prism.color.value import DEFAULT_SEED, value_channel
from prism.design.brush import Brush, apply_brushstroke, make_brush
from prism.design.style import PaintConfig, PainterlyStyle, style_config


DEFAULT_WORK_MAX_SIDE = 700
EPS = 1e-8

_STROKES = ("round", "flat", "filbert")


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
    pad: float = 0.06,
    seed: int = DEFAULT_SEED,
) -> PaintResult:
    check_rgb(rgb)

    if style == "acrylic" and config is None:
        from prism.design.segment import paint_segments

        return paint_segments(rgb, work_size=work_size, pad=pad, seed=seed)

    if pad > 0:
        p = int(round(pad * max(rgb.shape[:2])))
        if p > 0:
            rgb = np.pad(rgb, ((p, p), (p, p), (0, 0)), constant_values=1.0)

    cfg = style_config(style) if config is None else config
    rng = np.random.default_rng(seed)

    source, scale = resize_max_side(rgb, work_size)
    source = adjust_saturation(source, _cfg(cfg, "saturation", 1.0))

    canvas = initial_canvas(source, _cfg(cfg, "background", "mean"))
    strokes: list[BrushStroke] = []

    brush_cache: dict[object, Brush] = {}

    for radius in cfg.radii:
        ref = blurred_reference(source, radius, _cfg(cfg, "blur_factor", 0.5))
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

        for j, stroke in enumerate(layer):
            apply_stroke(
                canvas=canvas,
                stroke=stroke,
                config=cfg,
                brush_cache=brush_cache,
                seed=seed + 10_000 * int(radius) + j,
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

    image = np.clip(image, 0.0, 1.0).astype("float32")
    if _cfg(cfg, "background", "mean") == "white":
        image = dehaze(image)

    return PaintResult(
        image=image,
        strokes=out_strokes,
        style=style,
        config=cfg,
    )


def dehaze(
    image: np.ndarray,
    white_point: float = 0.85,
    soft: float = 0.10,
) -> np.ndarray:
    mn = image.min(axis=2)
    t = np.clip((mn - white_point) / max(soft, EPS), 0.0, 1.0)[..., None]
    return (image * (1.0 - t) + t).astype("float32")


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
    step = max(1, round(_cfg(config, "grid_factor", 1.0) * radius))

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
    config: PaintConfig,
    brush_cache: dict[object, Brush],
    seed: int,
) -> None:
    brush_type = _cfg(config, "brush", "simple")

    if brush_type == "bristle" and len(stroke.path) > 1:
        apply_bristle_stroke(
            canvas=canvas,
            stroke=stroke,
            config=config,
            brush_cache=brush_cache,
            seed=seed,
        )
        return

    if len(stroke.path) == 1:
        apply_dot(canvas, stroke)
    else:
        apply_polyline(canvas, stroke)


def apply_bristle_stroke(
    canvas: np.ndarray,
    stroke: BrushStroke,
    config: PaintConfig,
    brush_cache: dict[object, Brush],
    seed: int,
) -> None:
    vary = int(_cfg(config, "bristle_vary", 0))
    radius_round = max(1, int(round(stroke.radius)))

    if vary > 0:
        bucket = (seed // 7) % vary
        key: object = (radius_round, bucket)
    else:
        bucket = -1
        key = radius_round

    brush = brush_cache.get(key)
    if brush is None:
        if vary > 0:
            vr = np.random.default_rng(10_000 + bucket)
            rad = max(1.5, float(stroke.radius) * float(vr.uniform(0.8, 1.25)))
            cmin = max(4, int(round(stroke.radius * 2.0)))
            cmax = max(cmin + 4, int(round(stroke.radius * 5.0)))
            n_bristles = int(vr.integers(cmin, cmax + 1))
            shape = _STROKES[bucket % len(_STROKES)]
            brush_seed = 50_000 + bucket
        else:
            rad = float(stroke.radius)
            n_bristles = int(_cfg(config, "bristle_count", 48))
            shape = _cfg(config, "brush_shape", "filbert")
            brush_seed = seed

        brush = make_brush(
            radius=rad,
            n_bristles=n_bristles,
            shape=shape,
            seed=brush_seed,
            L_rng=_cfg(config, "bristle_length", (1.2, 2.2)),
            rho_rng=_cfg(config, "bristle_rigidity", (0.10, 0.28)),
            r_rng=_cfg(config, "bristle_width", (0.7, 1.8)),
            a_rng=_cfg(config, "bristle_alpha", (0.55, 1.00)),
            eta_rng=_cfg(config, "bristle_memory", (0.02, 0.16)),
        )
        brush_cache[key] = brush

    path = np.asarray(stroke.path, dtype="float32")
    color = np.asarray(stroke.color, dtype="float32")

    h, w = canvas.shape[:2]
    reach = float(brush.L.max() + np.abs(brush.xy).max() + brush.r.max()) + 2.0
    pad = int(np.ceil(reach))

    x0 = max(0, int(np.floor(float(path[:, 0].min()))) - pad)
    x1 = min(w, int(np.ceil(float(path[:, 0].max()))) + pad + 1)
    y0 = max(0, int(np.floor(float(path[:, 1].min()))) - pad)
    y1 = min(h, int(np.ceil(float(path[:, 1].max()))) + pad + 1)

    if x1 <= x0 or y1 <= y0:
        return

    sub = canvas[y0:y1, x0:x1]
    local_path = path - np.array([x0, y0], dtype="float32")

    out, _ = apply_brushstroke(
        canvas=sub,
        path=local_path,
        color=color,
        brush=brush,
        alpha=float(_cfg(config, "paint_opacity", stroke.alpha)),
        step=float(_cfg(config, "brush_step", 1.5)),
        samples=int(_cfg(config, "brush_samples", 12)),
        drag=float(_cfg(config, "brush_drag", 0.08)),
        dryness=float(_cfg(config, "dryness", 0.18)),
        grain=float(_cfg(config, "grain", 0.12)),
        rotate=bool(_cfg(config, "rotate_brush", False)),
        render=_cfg(config, "bristle_render", "splat"),
        seed=seed,
    )

    canvas[y0:y1, x0:x1] = out


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


def _cfg(
    config: PaintConfig,
    name: str,
    default,
):
    return getattr(config, name, default)


__all__ = [
    "BrushStroke",
    "PaintResult",
    "adjust_saturation",
    "apply_bristle_stroke",
    "blurred_reference",
    "color_error",
    "curved_path",
    "initial_canvas",
    "orientation_map",
    "paint",
    "resize_max_side",
    "stroke_color",
]
