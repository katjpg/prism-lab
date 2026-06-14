from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from prism.layer.paint import PaintLayer
from prism.layer.raster import RasterLayer

if TYPE_CHECKING:
    from prism.canvas.reference import Reference
    from prism.paint.style import BrushStyle
    from prism.raster.style import RasterStyle


@dataclass(frozen=True, slots=True)
class PaintTool:
    """Builds paint layers from a brush style.

    Parameters
    ----------
    style : BrushStyle
        Brush style applied to layers this tool creates.
    underpaint : str or None, optional
        Pigment name for an underpainting base (acrylic/gouache only).
    """

    style: "BrushStyle"
    underpaint: str | None = None

    def layer(self, source: "Reference | None" = None) -> PaintLayer:
        """Create a ``PaintLayer`` for ``source`` with this tool's style."""
        return PaintLayer(style=self.style, source=source, underpaint=self.underpaint)


@dataclass(frozen=True, slots=True)
class RasterTool:
    """Builds raster layers from a raster style.

    Parameters
    ----------
    style : RasterStyle
        Raster style applied to layers this tool creates.
    """

    style: "RasterStyle"

    def layer(self, source: "Reference | None" = None) -> RasterLayer:
        """Create a ``RasterLayer`` for ``source`` with this tool's style."""
        return RasterLayer(style=self.style, source=source)


__all__ = ["PaintTool", "RasterTool"]
