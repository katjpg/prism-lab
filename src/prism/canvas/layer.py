from __future__ import annotations

from dataclasses import dataclass

from prism.canvas.composite import BlendMode
from prism.layer.adjustment import AdjustmentLayer
from prism.layer.base import Layer


@dataclass
class LayerRecord:
    """Placement and compositing settings for a layer in a canvas stack.

    Parameters
    ----------
    layer : Layer or AdjustmentLayer
        Layer stored in this record.
    opacity : float
        Layer opacity in ``0..1``.
    blend : BlendMode
        Blend mode used to composite this layer over those below.
    visible : bool
        Whether to draw this layer.
    """

    layer: Layer | AdjustmentLayer
    opacity: float = 1.0
    blend: BlendMode = "normal"
    visible: bool = True


__all__ = ["LayerRecord"]
