from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from prism.layer.base import to_rgb
from prism.preset import Detail
from prism.raster.geometric import fit_primitives
from prism.raster.region import region_map
from prism.raster.stipple import stipple
from prism.raster.style import RasterStyle
from prism.raster.tonal import value_map

if TYPE_CHECKING:
    from prism.canvas.reference import Reference


@dataclass
class RasterLayer:
    """Redraws a source image using a raster style.

    Parameters
    ----------
    style : RasterStyle
        Raster style configuration.
    source : Reference or None, optional
        Image to redraw. Uses the canvas reference when ``None``.
    """

    style: RasterStyle
    source: "Reference | None" = None

    def contribution(
        self, src: np.ndarray, seed: int, detail: Detail
    ) -> tuple[np.ndarray, np.ndarray]:
        """Redraw ``src`` and return the ``(rgb, alpha)`` result."""
        params = self.style.params()
        mode = self.style.mode

        match mode:
            case "posterize":
                raw = value_map(
                    src,
                    n_bands=int(params["levels"]),
                    method=str(params["method"]),  # type: ignore[arg-type]
                    chroma=bool(params["perceived_value"]),
                    flatten=bool(params["even_lighting"]),
                    smooth=float(params["smoothness"]),
                    fill=str(params["fill"]),  # type: ignore[arg-type]
                    seed=seed,
                )
            case "segment":
                raw = region_map(
                    src,
                    fill=str(params["fill"]),  # type: ignore[arg-type]
                    method=str(params["method"]),  # type: ignore[arg-type]
                    n_segments=int(params["regions"]),
                    compactness=float(params["edge_fit"]),
                    smooth=float(params["smoothness"]),
                    markers=int(params["markers"]),
                    watershed_compactness=float(params["watershed_tightness"]),
                    detail=detail,
                    pixels=None,
                )
            case "stipple":
                raw = stipple(
                    src,
                    n_points=int(params["dots"]),
                    detail=detail,
                    pixels=None,
                    gamma=float(params["darkness"]),
                    smooth=float(params["smoothness"]),
                    dot_radius=float(params["dot_size"]),
                    edge_weight=float(params["edges"]),
                    white_cutoff=float(params["paper_white"]),
                    seed=seed,
                )
            case "geometric":
                raw = fit_primitives(
                    src,
                    n_shapes=int(params["shapes"]),
                    shape_family=str(params["shape_type"]),  # type: ignore[arg-type]
                    detail=detail,
                    pixels=None,
                    restarts=int(params["search"]),
                    edge_weight=float(params["edges"]),
                    residual_weight=float(params["image_fit"]),
                    alpha=float(params["opacity"]),
                    min_effort=float(params["effort"]),
                    seed=seed,
                )
            case _:
                raise ValueError(f"unknown raster mode: {mode!r}")

        image = to_rgb(raw.image)
        return image, np.ones(image.shape[:2], dtype="float32")


__all__ = ["RasterLayer"]
