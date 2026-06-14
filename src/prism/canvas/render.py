from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from prism.canvas.composite import composite
from prism.layer.adjustment import AdjustmentLayer
from prism.preset import Detail

if TYPE_CHECKING:
    from prism.canvas.layer import LayerRecord
    from prism.canvas.reference import Reference
    from prism.canvas.style import Ground


@dataclass(frozen=True, slots=True)
class Render:
    """Settings for rendering a canvas.

    Parameters
    ----------
    detail : {'draft', 'standard', 'high', 'ultra'}
        Quality preset for layer engines.
    size : tuple[int, int] or None, optional
        Output ``(width, height)``. Uses the native size when ``None``.
    dpi : int
        Resolution written when the artwork is saved.
    seed : int
        Seed used for reproducible rendering.
    """

    detail: Detail = "standard"
    size: tuple[int, int] | None = None
    dpi: int = 300
    seed: int = 7


@dataclass(frozen=True, slots=True)
class Artwork:
    """Rendered artwork.

    Attributes
    ----------
    rgb : np.ndarray
        Image of shape ``(H, W, 3)``, float32, range ``0..1``.
    dpi : int
        Resolution written to image metadata on save.
    """

    rgb: np.ndarray
    dpi: int = 300

    def save(self, path: str) -> None:
        """Save the artwork to ``path``.

        The file format is inferred from the extension.
        """
        save_image(self.rgb, path, dpi=self.dpi)


def compose(
    reference: Reference | None,
    records: list[LayerRecord],
    ground: Ground,
    output: Render,
) -> np.ndarray:
    """Composite visible layers from bottom to top into one image.

    Parameters
    ----------
    reference : Reference or None
        Default source for layers that do not have their own.
    records : list[LayerRecord]
        Layer records in bottom-to-top order.
    ground : Ground
        Opaque base the layers composite over.
    output : Render
        Rendering settings.

    Returns
    -------
    np.ndarray
        Composited image, shape ``(H, W, 3)``, float32, range ``0..1``.

    Raises
    ------
    ValueError
        If there is no reference image or no image-producing layer.
    """
    if reference is None:
        raise ValueError("canvas has no reference image")

    src = reference.rgb
    accum: np.ndarray | None = None

    for record in records:
        if not record.visible:
            continue
        layer = record.layer
        if isinstance(layer, AdjustmentLayer):
            if accum is not None:
                accum = layer.apply(accum)
            continue
        layer_src = layer.source.rgb if layer.source is not None else src
        rgb, alpha = layer.contribution(layer_src, output.seed, output.detail)
        if accum is None:
            # the first image layer sets the canvas resolution
            accum = ground.base(rgb, source=src)
        elif rgb.shape[:2] != accum.shape[:2]:
            # later layers (which may use a different engine/detail) are
            # resized to the canvas so stacks of mixed styles compose
            h, w = accum.shape[:2]
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        accum = composite(accum, rgb, alpha, mode=record.blend, opacity=record.opacity)

    if accum is None:
        raise ValueError("canvas has no image-producing layer")

    accum = np.clip(accum, 0.0, 1.0).astype("float32")
    if output.size is not None:
        width, height = output.size
        accum = cv2.resize(accum, (width, height), interpolation=cv2.INTER_AREA)
    return accum


def save_image(image: np.ndarray, path: str, dpi: int | None = None) -> None:
    """Save a ``0..1`` float RGB image to ``path`` as an 8-bit image.

    If ``dpi`` is given, write it to the image metadata.
    """
    arr = np.clip(image, 0.0, 1.0)
    arr = (arr * 255).round().astype("uint8")
    im = Image.fromarray(arr, mode="RGB")
    if dpi is None:
        im.save(path)
    else:
        im.save(path, dpi=(dpi, dpi))


__all__ = ["Artwork", "Render", "compose", "save_image"]
