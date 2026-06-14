from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from prism.layer.base import to_rgb
from prism.paint.engine import paint
from prism.paint.mask import bristle_stroke_mask
from prism.paint.region import paint_regions
from prism.paint.style import BrushStyle, build_paint_config
from prism.paint.underpaint import PIGMENTS, Underpainting
from prism.preset import Detail

if TYPE_CHECKING:
    from prism.canvas.reference import Reference

REGION_MODES = ("acrylic", "gouache")

ENGINE_STYLE = {
    "impressionism": "impressionist",
    "expressionism": "expressionist",
    "colorist": "colorist",
    "pointillism": "pointillist",
}


@dataclass
class PaintLayer:
    """Paints an image using a selected brush style.

    Parameters
    ----------
    style : BrushStyle
        Brush style configuration.
    source : Reference or None, optional
        Source image to paint. Uses the canvas reference when ``None``.
    underpaint : str or Underpainting or None, optional
        A pigment name (shorthand) or a full ``Underpainting`` config laid down
        before painting. Supported only for acrylic and gouache styles.
    """

    style: BrushStyle
    source: "Reference | None" = None
    underpaint: "str | Underpainting | None" = None

    def _underpainting(self) -> Underpainting | None:
        if self.underpaint is None or isinstance(self.underpaint, Underpainting):
            return self.underpaint
        if self.underpaint not in PIGMENTS:
            raise ValueError(f"unknown pigment: {self.underpaint!r}")
        return Underpainting(pigment=self.underpaint)

    def contribution(
        self, src: np.ndarray, seed: int, detail: Detail
    ) -> tuple[np.ndarray, np.ndarray]:
        """Paint ``src`` and return its ``(rgb, alpha)`` contribution.

        Acrylic and gouache report real coverage (transparent where unpainted);
        underpainting grounds and the streamline styles are opaque.
        """
        params = self.style.params()
        mode = self.style.mode

        if mode in REGION_MODES:
            footprint = bristle_stroke_mask if mode == "gouache" else None
            underpainting = self._underpainting()
            result = paint_regions(
                src,
                detail=detail,
                pixels=None,
                pad=float(params["border"]),
                seed=seed,
                compactness=float(params["edge_fit"]),
                coverage=float(params["coverage"]),
                color_jitter=float(params["color_variation"]),
                saturation=float(params["saturation"]),
                background=str(params["background"]),
                palette_colors=int(params["palette_colors"]),
                palette_mix=float(params["palette_mix"]),
                color_smooth=float(params["color_smooth"]),
                palette_saturation=float(params["color_sat"]),
                underpainting=underpainting,
                footprint=footprint,
                transparent=underpainting is None,
            )
        else:
            if self.underpaint is not None:
                raise ValueError("underpaint is supported on acrylic/gouache only")
            engine_style = ENGINE_STYLE[mode]
            config = build_paint_config(engine_style, mode, params)
            result = paint(
                src,
                style=engine_style,  # type: ignore[arg-type]
                config=config,
                detail=detail,
                pixels=None,
                pad=float(params["border"]),
                seed=seed,
            )

        image = to_rgb(result.image)
        if result.alpha is not None:
            return image, result.alpha
        return image, np.ones(image.shape[:2], dtype="float32")


__all__ = ["PaintLayer"]
