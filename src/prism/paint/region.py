from __future__ import annotations

import hashlib
import os
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import regionprops

from prism.color.value import DEFAULT_SEED, adjust_saturation, check_rgb
from prism.paint.ground import (
    PAINT_BORDER,
    blend_texture,
    create_canvas,
    reflect_border,
)
from prism.paint.brush import Stroke, accumulate_coverage, render_tiled
from prism.paint.style import PaintResult, RegionPaintConfig
from prism.paint.mask import (
    MaskBuilder,
    TemplateCache,
    create_soft_mask_template,
    get_bristle_cache,
    get_soft_cache,
)
from prism.paint.underpaint import (
    Underpainting,
    darken_edges,
    value_to_underpaint,
)
from prism.preset import Detail, fit_image
from prism.raster.region import superpixels

EPS = 1e-8

PATH_STEP = 1.5
STAMP_TRAIL = 0.6
STAMP_WIDTH_SCALE = 1.8

RADIUS_MIN = 1.0
RADIUS_MAX = 26.0

COMPACTNESS = 2.2
MIN_REGION_AREA = 4

UNDERPAINT_DETAIL = "standard"
UNDERPAINT_TRAIL = 0.5
UNDERPAINT_WIDTH_SCALE = 2.0
UNDERPAINT_GROW_SCALE = 0.35

TRANSPARENT_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class RegionPass:
    n_seg: int
    radius: float
    alpha: float


REGION_PASSES: tuple[RegionPass, ...] = (
    RegionPass(160, 11.0, 0.92),
    RegionPass(420, 8.2, 0.84),
    RegionPass(900, 5.8, 0.72),
    RegionPass(1800, 4.0, 0.60),
    RegionPass(3200, 2.8, 0.50),
    RegionPass(5200, 1.9, 0.40),
    RegionPass(7600, 1.35, 0.32),
)


DETAIL_PASSES: dict[Detail, tuple[int, ...]] = {
    "draft": (0, 1, 2, 3),
    "standard": (0, 1, 2, 3, 4),
    "high": (0, 1, 2, 3, 4, 5),
    "ultra": tuple(range(len(REGION_PASSES))),
}


@dataclass(frozen=True, slots=True)
class LabelOrientation:
    theta: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True, slots=True)
class LabelGroups:
    order: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    labels: np.ndarray


@dataclass(slots=True)
class RegionGeometry:
    labels: np.ndarray
    regions: list
    theta_map: np.ndarray
    coherence: np.ndarray
    label_orient: LabelOrientation | None = None
    label_groups: LabelGroups | None = None


def tonal_color_bias(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[..., None]

    shadow = np.clip((0.55 - gray) / 0.55, 0.0, 1.0)
    light = np.clip((gray - 0.55) / 0.45, 0.0, 1.0)

    out = x.copy()
    out[..., 2] += 0.025 * shadow[..., 0]
    out[..., 1] += 0.010 * shadow[..., 0]
    out[..., 0] += 0.010 * light[..., 0]
    out[..., 1] += 0.005 * light[..., 0]

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def quantize_lab_kmeans(
    rgb: np.ndarray,
    colors: int,
    seed: int,
    max_samples: int = 160_000,
) -> np.ndarray:
    x = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    if colors <= 1:
        return x

    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    h, w = lab.shape[:2]
    flat = lab.reshape(-1, 3).astype(np.float32)

    rng = np.random.default_rng(seed)
    if len(flat) > max_samples:
        idx = rng.choice(len(flat), size=max_samples, replace=False)
        train = flat[idx]
    else:
        train = flat

    k = int(min(max(2, colors), len(train)))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        35,
        0.5,
    )

    cv2.setRNGSeed(int(seed))
    _, _, centers = cv2.kmeans(
        train,
        k,
        None,  # type: ignore[arg-type]
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = np.empty(len(flat), dtype=np.int32)
    chunk = 200_000

    for i in range(0, len(flat), chunk):
        j = min(len(flat), i + chunk)
        d = flat[i:j, None, :] - centers[None, :, :]
        labels[i:j] = np.argmin(np.sum(d * d, axis=2), axis=1)

    quant = centers[labels].reshape(h, w, 3).astype(np.float32)
    out = cv2.cvtColor(quant, cv2.COLOR_LAB2RGB)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def create_color_source(
    rgb: np.ndarray,
    *,
    smooth: float,
    palette_saturation: float,
    colors: int,
    mix: float,
    seed: int,
    max_samples: int = 160_000,
) -> np.ndarray:
    x = cv2.GaussianBlur(
        rgb.astype(np.float32),
        ksize=(0, 0),
        sigmaX=max(0.0, float(smooth)),
    )

    x = adjust_saturation(x, palette_saturation)
    x = tonal_color_bias(x)

    if colors > 1 and mix > 0:
        q = quantize_lab_kmeans(x, colors, seed, max_samples=max_samples)
        x = (1.0 - float(mix)) * x + float(mix) * q

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def region_color(
    color_source: np.ndarray,
    coords: np.ndarray | None,
    color_jitter: float,
    rng: np.random.Generator,
    base: np.ndarray | None = None,
) -> np.ndarray:
    if base is None:
        assert coords is not None
        pixels = color_source[coords[:, 0], coords[:, 1]].astype(np.float32)
        base = np.median(pixels, axis=0).astype(np.float32)

    color = np.asarray(base, dtype=np.float32).copy()

    if color_jitter > 0:
        gain = 1.0 - color_jitter / 2.0 + color_jitter * rng.random(3)
        color = color * gain.astype(np.float32)

    return np.clip(color, 0.0, 1.0).astype(np.float32)


def label_groups(labels: np.ndarray) -> LabelGroups:
    lab = labels.ravel().astype(np.int32)
    order = np.argsort(lab, kind="stable")
    lab_sorted = lab[order]

    starts = np.r_[0, np.flatnonzero(np.diff(lab_sorted)) + 1]
    ends = np.r_[starts[1:], len(lab_sorted)]

    return LabelGroups(
        order=order,
        starts=starts,
        ends=ends,
        labels=lab_sorted[starts].astype(np.int32),
    )


def label_median_colors(groups: LabelGroups, color_source: np.ndarray) -> np.ndarray:
    colors = color_source.reshape(-1, 3).astype(np.float32)[groups.order]
    n_lab = int(groups.labels[-1]) + 1
    med = np.zeros((n_lab, 3), dtype=np.float32)

    for start, end, lab in zip(groups.starts, groups.ends, groups.labels):
        med[int(lab)] = np.median(colors[start:end], axis=0)

    return med.astype(np.float32)


def structure_tensor_field(
    source: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(source.astype(np.float32), cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma)

    grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    theta = grad + np.pi / 2.0

    num = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    den = jxx + jyy + EPS
    coh = np.clip(num / den, 0.0, 1.0)

    return theta.astype(np.float32), coh.astype(np.float32)


def region_orientation(
    region,
    theta_map: np.ndarray,
    coherence: np.ndarray,
    *,
    min_confidence: float = 0.08,
) -> float:
    coords = region.coords
    ys = coords[:, 0]
    xs = coords[:, 1]

    a = theta_map[ys, xs].astype(np.float32)
    w = coherence[ys, xs].astype(np.float32)

    if float(w.mean()) < min_confidence:
        if float(region.eccentricity) > 0.82:
            o = float(region.orientation)
            return float(np.arctan2(np.cos(o), -np.sin(o)))
        return 0.0

    s = float(np.sum(w * np.sin(2.0 * a)))
    c = float(np.sum(w * np.cos(2.0 * a)))
    return 0.5 * float(np.arctan2(s, c))


def label_orientation(
    labels: np.ndarray,
    theta_map: np.ndarray,
    coherence: np.ndarray,
) -> LabelOrientation:
    lab = labels.ravel().astype(np.int32)
    n_lab = int(lab.max()) + 1

    a = theta_map.ravel().astype(np.float32)
    w = coherence.ravel().astype(np.float32)

    area = np.bincount(lab, minlength=n_lab).astype(np.float32)
    sw = np.bincount(lab, weights=w.astype(np.float64), minlength=n_lab)
    ss = np.bincount(lab, weights=(w * np.sin(2.0 * a)).astype(np.float64), minlength=n_lab)
    cc = np.bincount(lab, weights=(w * np.cos(2.0 * a)).astype(np.float64), minlength=n_lab)

    theta = 0.5 * np.arctan2(ss, cc)
    conf = sw / np.maximum(area, 1.0)

    return LabelOrientation(
        theta=theta.astype(np.float32),
        confidence=conf.astype(np.float32),
    )


def label_theta(
    region,
    label_orient: LabelOrientation,
    min_confidence: float = 0.08,
) -> float:
    lab = int(region.label)

    if float(label_orient.confidence[lab]) < min_confidence:
        if float(region.eccentricity) > 0.82:
            o = float(region.orientation)
            return float(np.arctan2(np.cos(o), -np.sin(o)))
        return 0.0

    return float(label_orient.theta[lab])


_GATE_KERNELS: dict[int, np.ndarray] = {}


def _gate_kernel(grow: int) -> np.ndarray:
    k = 2 * int(grow) + 1
    kernel = _GATE_KERNELS.get(k)
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        _GATE_KERNELS[k] = kernel
    return kernel


def region_gate(
    region,
    shape_hw: tuple[int, int],
    *,
    pad_px: int,
    top: int,
    left: int,
    grow: int,
) -> np.ndarray:
    h, w = shape_hw
    out = np.zeros((h, w), dtype=np.uint8)

    minr, minc, _, _ = region.bbox
    rmask = region.image

    oy = minr + pad_px - top
    ox = minc + pad_px - left

    y0 = max(0, oy)
    x0 = max(0, ox)
    y1 = min(h, oy + rmask.shape[0])
    x1 = min(w, ox + rmask.shape[1])

    if y1 > y0 and x1 > x0:
        sub = rmask[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
        out[y0:y1, x0:x1][sub] = 255

    if grow > 0:
        out = cv2.dilate(out, _gate_kernel(grow), iterations=1)

    return (out.astype(np.float32) / 255.0).astype(np.float32)


def render_region_stroke(
    region,
    color_source: np.ndarray,
    theta_map: np.ndarray,
    coherence: np.ndarray,
    *,
    cfg: RegionPass,
    coverage: float,
    alpha: float,
    color_jitter: float,
    footprint: MaskBuilder,
    pad_px: int,
    h: int,
    w: int,
    seed: int,
    label_colors: np.ndarray | None = None,
    label_orient: LabelOrientation | None = None,
    trail: float = STAMP_TRAIL,
    width_scale: float = STAMP_WIDTH_SCALE,
    grow_scale: float = 1.0,
) -> Stroke | None:
    rng = np.random.default_rng(seed)

    radius = float(
        np.clip(
            max(cfg.radius, 0.40 * float(region.axis_minor_length)) * coverage,
            RADIUS_MIN,
            RADIUS_MAX,
        )
    )

    if label_orient is not None:
        theta = label_theta(region, label_orient)
    else:
        theta = region_orientation(region, theta_map, coherence)

    major = max(float(region.axis_major_length), 2.0)
    half_len = max(0.5 * min(0.40 * major, radius * 1.65) * coverage, PATH_STEP * 1.5)
    length = 2.0 * half_len + 2.0 * trail * radius
    width = width_scale * radius

    foot, fcy, fcx = footprint(length, width, float(np.degrees(theta)))
    fh, fw = foot.shape

    cy, cx = region.centroid
    top = int(round(float(cy) + pad_px - fcy))
    left = int(round(float(cx) + pad_px - fcx))

    if top + fh <= 0 or left + fw <= 0 or top >= h or left >= w:
        return None

    grow = max(1, int(round(radius * 0.85 * PAINT_BORDER * grow_scale)))
    gate = region_gate(region, (fh, fw), pad_px=pad_px, top=top, left=left, grow=grow)

    opacity = float(alpha * cfg.alpha)
    a = np.clip(foot * gate * opacity, 0.0, 1.0).astype(np.float32)
    if float(a.max()) <= 0.0:
        return None

    if label_colors is not None:
        base = label_colors[int(region.label)]
        color = region_color(color_source, None, color_jitter, rng, base=base)
    else:
        color = region_color(color_source, region.coords, color_jitter, rng)

    return Stroke(alpha=a, color=color, top=top, left=left)


def worker_count(n: int) -> int:
    if n > 0:
        return n
    return min(8, os.cpu_count() or 1)


def paint_regions(
    rgb: np.ndarray,
    detail: Detail = "standard",
    *,
    compactness: float | None = 5.0,
    coverage: float = 1.6,
    alpha: float = 1.0,
    color_jitter: float = 0.004,
    saturation: float = 1.06,
    background: str = "white",
    pad: float = 0.05,
    underpainting: Underpainting | None = None,
    palette_colors: int = 16,
    palette_mix: float = 0.42,
    color_smooth: float = 1.5,
    palette_saturation: float = 1.15,
    geometry_cache: MutableMapping | None = None,
    workers: int = 1,
    footprint: MaskBuilder | None = None,
    transparent: bool = False,
    stroke_trail: float = STAMP_TRAIL,
    stroke_width_scale: float = STAMP_WIDTH_SCALE,
    gate_grow_scale: float = 1.0,
    seed: int = DEFAULT_SEED,
    pixels: tuple[int, int] | None = None,
) -> PaintResult:
    check_rgb(rgb)

    src, scale = fit_image(rgb, detail, pixels)
    src = adjust_saturation(src, saturation)

    color_source = create_color_source(
        src,
        smooth=color_smooth,
        palette_saturation=palette_saturation,
        colors=palette_colors,
        mix=palette_mix,
        seed=seed,
    )

    pad_px = int(round(pad * max(src.shape[:2]))) if pad > 0 else 0
    h, w = src.shape[:2]

    cov = None
    if transparent:
        canvas = np.zeros((h + 2 * pad_px, w + 2 * pad_px, 3), dtype=np.float32)
        cov = np.zeros(canvas.shape[:2], dtype=np.float32)
    elif underpainting is not None:
        canvas = render_underpaint(
            src,
            underpainting,
            background,
            seed,
            pad,
            (h + 2 * pad_px, w + 2 * pad_px),
        )
    else:
        base = create_canvas(src, background)
        canvas = reflect_border(base, pad_px) if pad_px > 0 else base

    ch, cw = canvas.shape[:2]
    c = float(compactness) if compactness is not None else COMPACTNESS
    n_workers = worker_count(workers)
    rng = np.random.default_rng(seed)

    image_id = (
        hashlib.blake2b(
            np.ascontiguousarray(src).tobytes(),
            digest_size=16,
        ).hexdigest()
        if geometry_cache is not None
        else None
    )

    if footprint is None:
        footprint = get_bristle_cache().get

    for pidx in DETAIL_PASSES[detail]:
        cfg = REGION_PASSES[pidx]

        n_seg = cfg.n_seg

        if geometry_cache is not None:
            geo_key = (image_id, c, n_seg, cfg.radius, pidx)
            geo = geometry_cache.get(geo_key)
        else:
            geo_key = None
            geo = None

        if geo is None:
            labels = superpixels(src, n_segments=n_seg, compactness=c)
            regions = [r for r in regionprops(labels) if r.area >= MIN_REGION_AREA]

            if not regions:
                continue

            blurred = cv2.GaussianBlur(
                src.astype(np.float32),
                ksize=(0, 0),
                sigmaX=max(1.0, cfg.radius * 0.50),
            )
            theta_map, coherence = structure_tensor_field(
                blurred,
                sigma=max(2.0, cfg.radius * 0.35),
            )

            geo = RegionGeometry(labels, regions, theta_map, coherence)
            if geometry_cache is not None:
                geometry_cache[geo_key] = geo

        labels = geo.labels
        regions = geo.regions
        theta_map = geo.theta_map
        coherence = geo.coherence
        label_orient = geo.label_orient
        groups = geo.label_groups

        if label_orient is None:
            label_orient = label_orientation(labels, theta_map, coherence)
            geo.label_orient = label_orient

        if groups is None:
            groups = label_groups(labels)
            geo.label_groups = groups
        label_colors = label_median_colors(groups, color_source)

        def work(
            region,
            _theta=theta_map,
            _coh=coherence,
            _cfg=cfg,
            _pidx=pidx,
            _footprint=footprint,
            _label_colors=label_colors,
            _label_orient=label_orient,
        ) -> Stroke | None:
            rseed = seed + 1_000_003 * _pidx + 9_176 * int(region.label)
            return render_region_stroke(
                region,
                color_source,
                _theta,
                _coh,
                cfg=_cfg,
                coverage=coverage,
                alpha=alpha,
                color_jitter=color_jitter,
                footprint=_footprint,
                pad_px=pad_px,
                h=ch,
                w=cw,
                seed=rseed,
                label_colors=_label_colors,
                label_orient=_label_orient,
                trail=stroke_trail,
                width_scale=stroke_width_scale,
                grow_scale=gate_grow_scale,
            )

        if n_workers > 1 and len(regions) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                strokes = list(pool.map(work, regions))
        else:
            strokes = [work(region) for region in regions]

        strokes = [s for s in strokes if s is not None]
        rng.shuffle(strokes)
        render_tiled(canvas, strokes)
        if cov is not None:
            accumulate_coverage(cov, strokes)

    config = RegionPaintConfig(
        coverage=coverage,
        compactness=c,
        alpha=alpha,
        saturation=saturation,
        background=background,
        color_jitter=color_jitter,
        palette_colors=palette_colors,
        palette_mix=palette_mix,
        color_smooth=color_smooth,
        palette_saturation=palette_saturation,
    )

    if cov is not None:
        image, layer_alpha = _recover_straight_rgba(canvas, cov, scale)
        return PaintResult(
            image=image,
            style="regions",
            config=config,
            detail=detail,
            alpha=layer_alpha,
        )

    if scale != 1.0:
        ch, cw = canvas.shape[:2]
        image = cv2.resize(
            canvas,
            (round(cw * scale), round(ch * scale)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
    else:
        image = canvas.astype(np.float32)

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    image = blend_texture(image, background)

    return PaintResult(
        image=image,
        style="regions",
        config=config,
        detail=detail,
    )


def _recover_straight_rgba(
    premult: np.ndarray,
    coverage: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    # ``premult`` stores RGB composited over black, so recover straight RGB by
    # dividing by alpha only where the layer actually painted.
    if scale != 1.0:
        ch, cw = premult.shape[:2]
        size = (round(cw * scale), round(ch * scale))
        premult = cv2.resize(premult, size, interpolation=cv2.INTER_LINEAR)
        coverage = cv2.resize(coverage, size, interpolation=cv2.INTER_LINEAR)

    alpha = np.clip(coverage, 0.0, 1.0).astype(np.float32)
    painted = alpha[..., None] > TRANSPARENT_EPS
    rgb = np.where(
        painted,
        premult / np.maximum(alpha[..., None], TRANSPARENT_EPS),
        0.0,
    )
    return np.clip(rgb, 0.0, 1.0).astype(np.float32), alpha


def render_underpaint(
    source: np.ndarray,
    config: Underpainting,
    background: str,
    seed: int,
    pad: float,
    target_shape: tuple[int, int],
) -> np.ndarray:
    tonemap = darken_edges(
        value_to_underpaint(source, config.spec()),
        source,
        config.edge_amount,
        config.edge_width,
    )

    if config.softness == 2.0 and config.blur == 0.0:
        footprint = get_soft_cache().get
    else:
        footprint = TemplateCache(
            create_soft_mask_template(aa=config.softness, soften=config.blur)
        ).get

    under = paint_regions(
        tonemap,
        detail=UNDERPAINT_DETAIL,
        coverage=config.coverage,
        alpha=config.alpha,
        background=background,
        pad=pad,
        seed=seed,
        footprint=footprint,
        stroke_trail=UNDERPAINT_TRAIL,
        stroke_width_scale=UNDERPAINT_WIDTH_SCALE,
        gate_grow_scale=UNDERPAINT_GROW_SCALE,
    ).image

    if under.shape[:2] != tuple(target_shape):
        under = cv2.resize(under, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
    return under


__all__ = [
    "paint_regions",
    "render_underpaint",
]
