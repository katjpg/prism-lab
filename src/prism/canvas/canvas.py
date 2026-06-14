from __future__ import annotations

from dataclasses import dataclass, field

from prism.canvas.composite import BlendMode
from prism.canvas.layer import LayerRecord
from prism.canvas.reference import Reference
from prism.canvas.render import Artwork, Render, compose
from prism.canvas.style import Ground
from prism.layer.adjustment import AdjustmentLayer
from prism.layer.base import Layer


@dataclass
class Canvas:
    """Layer stack that renders to a single image.

    Parameters
    ----------
    reference : Reference or None, optional
        Default source image for layers that do not have their own.
    ground : Ground, optional
        Opaque base the layers composite over.
    """

    reference: Reference | None = None
    ground: Ground = field(default_factory=Ground)
    records: list[LayerRecord] = field(default_factory=list)

    def add(
        self,
        layer: Layer | AdjustmentLayer,
        *,
        opacity: float = 1.0,
        blend: BlendMode = "normal",
    ) -> LayerRecord:
        """Add ``layer`` to the top of the stack.

        Parameters
        ----------
        layer : Layer or AdjustmentLayer
            Layer to add.
        opacity : float, default=1.0
            Layer opacity in ``0..1``.
        blend : BlendMode, default='normal'
            Blend mode used when compositing the layer.

        Returns
        -------
        LayerRecord
            Record created for the added layer.
        """
        record = LayerRecord(layer=layer, opacity=opacity, blend=blend)
        self.records.append(record)
        return record

    def remove(self, record: LayerRecord) -> None:
        """Remove ``record`` from the stack."""

        self.records.remove(record)

    def move(self, record: LayerRecord, index: int) -> None:
        """Move ``record`` to ``index`` in the stack."""

        self.records.remove(record)
        self.records.insert(index, record)

    def render(self, output: Render = Render()) -> Artwork:
        """Render the canvas and return the resulting artwork.

        Parameters
        ----------
        output : Render, default=Render()
            Rendering settings.

        Returns
        -------
        Artwork
            Rendered artwork.
        """
        rgb = compose(self.reference, self.records, self.ground, output)
        return Artwork(rgb=rgb, dpi=output.dpi)


__all__ = ["Canvas"]
