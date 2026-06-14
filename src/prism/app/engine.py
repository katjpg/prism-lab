from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from prism.app.control import CategoryName, get_mode, resolve_params
from prism.color.adjust import ColorWheel, apply_color_wheel, hue_rotate
from prism.color.palette import RECOLOR_SCHEMES, Palette, recolor
from prism.color.value import Value, check_rgb
from prism.paint.engine import paint
from prism.paint.region import paint_regions
from prism.paint.style import PaintResult, build_paint_config
from prism.paint.mask import bristle_stroke_mask
from prism.preset import Detail
from prism.raster.geometric import PrimitiveFit, fit_primitives
from prism.raster.region import BoundaryMap, region_map
from prism.raster.stipple import StippleMap, stipple
from prism.raster.tonal import ValueMap, value_map


@dataclass(frozen=True, slots=True)
class RenderOptions:
    seed: int = 7
    detail: Detail = "standard"
    pixels: tuple[int, int] | None = None
    color: ColorWheel = ColorWheel()


@dataclass(frozen=True, slots=True)
class RenderRequest:
    rgb: np.ndarray
    category: CategoryName
    mode: str
    params: dict[str, Any]
    options: RenderOptions = RenderOptions()


@dataclass(frozen=True, slots=True)
class RenderResult:
    image: np.ndarray
    raw: Any
    category: str
    mode: str
    params: dict[str, Any]


ENGINE_STYLE = {
    "impressionism": "impressionist",
    "expressionism": "expressionist",
    "colorist": "colorist",
    "pointillism": "pointillist",
}

GEOMETRY_CACHE_MAX = 32
_GEOMETRY_CACHE_STORE: dict = {}
_GEOMETRY_CACHE_IMAGE: str | None = None


def _geometry_cache_for(rgb: np.ndarray) -> dict:
    global _GEOMETRY_CACHE_IMAGE
    img_id = hashlib.blake2b(np.ascontiguousarray(rgb).tobytes(), digest_size=16).hexdigest()
    if img_id != _GEOMETRY_CACHE_IMAGE:
        _GEOMETRY_CACHE_STORE.clear()
        _GEOMETRY_CACHE_IMAGE = img_id
    return _GEOMETRY_CACHE_STORE


def _prune_geometry_cache() -> None:
    while len(_GEOMETRY_CACHE_STORE) > GEOMETRY_CACHE_MAX:
        _GEOMETRY_CACHE_STORE.pop(next(iter(_GEOMETRY_CACHE_STORE)))


def render(request: RenderRequest) -> RenderResult:
    check_rgb(request.rgb)

    mode = get_mode(request.category, request.mode)
    params = resolve_params(mode, request.params)

    match request.category:
        case "paint":
            result = render_paint(request.rgb, request.mode, params, request.options)
        case "abstract":
            result = render_abstract(request.rgb, request.mode, params, request.options)
        case "color":
            result = render_color(request.rgb, request.mode, params, request.options)
        case _:
            raise ValueError(f"unknown category: {request.category!r}")

    image = extract_image(result, sort=str(params.get("sort", "lightness")))
    image = apply_color_wheel(image, request.options.color)

    return RenderResult(
        image=image,
        raw=result,
        category=request.category,
        mode=request.mode,
        params=params,
    )


def render_mode(
    rgb: np.ndarray,
    category: CategoryName,
    mode: str,
    params: dict[str, Any] | None = None,
    *,
    seed: int = 7,
    detail: Detail = "standard",
    pixels: tuple[int, int] | None = None,
    color: ColorWheel | None = None,
) -> RenderResult:
    return render(
        RenderRequest(
            rgb=rgb,
            category=category,
            mode=mode,
            params=params or {},
            options=RenderOptions(
                seed=seed,
                detail=detail,
                pixels=pixels,
                color=color or ColorWheel(),
            ),
        )
    )


def render_paint(
    rgb: np.ndarray,
    mode: str,
    params: dict[str, Any],
    options: RenderOptions,
) -> PaintResult:
    if mode in ("acrylic", "gouache"):
        footprint = bristle_stroke_mask if mode == "gouache" else None
        result = paint_regions(
            rgb,
            detail=options.detail,
            pixels=options.pixels,
            pad=float(params["border"]),
            seed=options.seed,
            compactness=float(params["edge_fit"]),
            coverage=float(params["coverage"]),
            color_jitter=float(params["color_variation"]),
            saturation=float(params["saturation"]),
            background=str(params["background"]),
            palette_colors=int(params["palette_colors"]),
            palette_mix=float(params["palette_mix"]),
            color_smooth=float(params["color_smooth"]),
            palette_saturation=float(params["color_sat"]),
            footprint=footprint,
            geometry_cache=_geometry_cache_for(rgb),
        )
        _prune_geometry_cache()
        return result

    engine_style = ENGINE_STYLE[mode]
    cfg = build_paint_config(engine_style, mode, params)

    return paint(
        rgb,
        style=engine_style,  # type: ignore[arg-type]
        config=cfg,
        detail=options.detail,
        pixels=options.pixels,
        pad=float(params["border"]),
        seed=options.seed,
    )


def render_abstract(
    rgb: np.ndarray,
    mode: str,
    params: dict[str, Any],
    options: RenderOptions,
) -> ValueMap | BoundaryMap | StippleMap | PrimitiveFit:
    if mode == "posterize":
        return render_posterize(rgb, params, options)

    if mode == "segment":
        return region_map(
            rgb,
            fill=str(params["fill"]),  # type: ignore[arg-type]
            method=str(params["method"]),  # type: ignore[arg-type]
            n_segments=int(params["regions"]),
            compactness=float(params["edge_fit"]),
            smooth=float(params["smoothness"]),
            markers=int(params["markers"]),
            watershed_compactness=float(params["watershed_tightness"]),
            detail=options.detail,
            pixels=options.pixels,
        )

    if mode == "stipple":
        return stipple(
            rgb,
            n_points=int(params["dots"]),
            detail=options.detail,
            pixels=options.pixels,
            gamma=float(params["darkness"]),
            smooth=float(params["smoothness"]),
            dot_radius=float(params["dot_size"]),
            edge_weight=float(params["edges"]),
            white_cutoff=float(params["paper_white"]),
            seed=options.seed,
        )

    if mode == "geometric":
        return fit_primitives(
            rgb,
            n_shapes=int(params["shapes"]),
            shape_family=str(params["shape_type"]),  # type: ignore[arg-type]
            detail=options.detail,
            pixels=options.pixels,
            restarts=int(params["search"]),
            edge_weight=float(params["edges"]),
            residual_weight=float(params["image_fit"]),
            alpha=float(params["opacity"]),
            min_effort=float(params["effort"]),
            seed=options.seed,
        )

    raise ValueError(f"unknown abstract mode: {mode!r}")


def render_posterize(
    rgb: np.ndarray,
    params: dict[str, Any],
    options: RenderOptions,
) -> ValueMap:
    return value_map(
        rgb,
        n_bands=int(params["levels"]),
        method=str(params["method"]),  # type: ignore[arg-type]
        chroma=bool(params["perceived_value"]),
        flatten=bool(params["even_lighting"]),
        smooth=float(params["smoothness"]),
        fill=str(params["fill"]),  # type: ignore[arg-type]
        seed=options.seed,
    )


def render_color(
    rgb: np.ndarray,
    mode: str,
    params: dict[str, Any],
    options: RenderOptions,
) -> Palette | np.ndarray:
    if mode == "palette":
        return Palette.extract(
            rgb,
            n_colors=int(params["colors"]),
            n_samples=int(params["precision"]),
            seed=options.seed,
            detail=options.detail,
            pixels=options.pixels,
        )

    if mode == "recolor":
        scheme = str(params["palette"])
        try:
            anchors = RECOLOR_SCHEMES[scheme]
        except KeyError as exc:
            raise ValueError(f"unknown recolor palette: {scheme!r}") from exc

        value = Value(n_bands=5, smooth=True, seed=options.seed).extract(rgb)

        return recolor(
            rgb,
            value=value,
            anchors=anchors,
            amount=float(params["strength"]),
            chroma=float(params["vibrance"]),
            sigma=float(params["smoothness"]),
        )

    raise ValueError(f"unknown color mode: {mode!r}")


def extract_image(result: Any, sort: str = "lightness") -> np.ndarray:
    if isinstance(result, Palette):
        return palette_image(result, sort=sort)

    if isinstance(result, np.ndarray):
        image = result.astype("float32")
    elif hasattr(result, "image"):
        image = result.image.astype("float32")
    else:
        raise TypeError(f"cannot extract image from result type: {type(result).__name__}")

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)

    return image


def palette_image(
    palette: Palette,
    sort: str = "lightness",
    height: int = 160,
    swatch_width: int = 120,
) -> np.ndarray:
    if sort == "weight":
        colors = palette.by_weight()
    else:
        colors = palette.by_lightness()

    width = max(1, swatch_width * len(colors))
    out = np.ones((height, width, 3), dtype="float32")

    for i, color in enumerate(colors):
        x0 = i * swatch_width
        x1 = x0 + swatch_width
        out[:, x0:x1, :] = np.asarray(color.rgb, dtype="float32")

    return out


__all__ = [
    "RECOLOR_SCHEMES",
    "CategoryName",
    "ColorWheel",
    "RenderOptions",
    "RenderRequest",
    "RenderResult",
    "apply_color_wheel",
    "build_paint_config",
    "extract_image",
    "hue_rotate",
    "palette_image",
    "render",
    "render_mode",
]
