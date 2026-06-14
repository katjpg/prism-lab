from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from prism.color.adjust import Adjustment
from prism.color.palette import Palette
from prism.color.value import check_rgb


@dataclass(frozen=True, slots=True)
class Crop:
    """Pixel crop box measured from the top-left corner.

    Parameters
    ----------
    left, top : int
        Top-left corner, in pixels.
    width, height : int
        Box size, in pixels.
    """

    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Reference:
    """Source image stored as normalized RGB.

    Attributes
    ----------
    rgb : np.ndarray
        Image of shape ``(H, W, 3)``, float32, range ``0..1``.
    """

    rgb: np.ndarray

    @classmethod
    def open(
        cls,
        path: str,
        *,
        crop: Crop | None = None,
        fit: tuple[int, int] | None = None,
    ) -> Reference:
        """Load an image file as a reference.

        Parameters
        ----------
        path : str
            Path to the image file.
        crop : Crop or None, optional
            Region to keep before scaling.
        fit : tuple[int, int] or None, optional
            ``(width, height)`` box to scale into while preserving aspect ratio.

        Returns
        -------
        Reference
            Loaded reference image.
        """
        image = Image.open(path).convert("RGB")
        if crop is not None:
            image = image.crop(
                (crop.left, crop.top, crop.left + crop.width, crop.top + crop.height)
            )
        if fit is not None:
            image.thumbnail(fit, Image.Resampling.LANCZOS)
        return cls.from_rgb(np.asarray(image, dtype="float32") / 255.0)

    @classmethod
    def from_rgb(cls, rgb: np.ndarray) -> Reference:
        """Create a reference from an ``(H, W, 3)`` float RGB array in ``0..1``."""
        arr = np.asarray(rgb, dtype="float32")
        check_rgb(arr)
        return cls(rgb=arr)

    def adjust(self, *steps: Adjustment) -> Reference:
        """Return a new reference with ``steps`` applied in order."""
        out = self.rgb
        for step in steps:
            out = step.apply(out)
        return Reference.from_rgb(out)

    def palette(self, colors: int = 8) -> Palette:
        """Extract an ordered palette of dominant colors.

        Parameters
        ----------
        colors : int, default=8
            Number of colors to extract.

        Returns
        -------
        Palette
            Extracted palette.
        """
        return Palette.extract(self.rgb, n_colors=colors)


__all__ = ["Crop", "Reference"]
