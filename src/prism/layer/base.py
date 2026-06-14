from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from prism.preset import Detail

if TYPE_CHECKING:
    from prism.canvas.reference import Reference


class Layer(Protocol):
    """Protocol for layers that render an image into RGB and alpha."""

    source: Reference | None

    def contribution(
        self, src: np.ndarray, seed: int, detail: Detail
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render ``src`` and return ``(rgb, alpha)``.

        Parameters
        ----------
        src : np.ndarray
            Source image.
        seed : int
            Seed for stochastic rendering.
        detail : {'draft', 'standard', 'high', 'ultra'}
            Rendering quality preset.

        Returns
        -------
        rgb : np.ndarray
            Rendered colors.
        alpha : np.ndarray
            Layer coverage.
        """
        ...


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as ``(H, W, 3)`` float32, repeating a grayscale plane."""
    img = image.astype("float32")
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    return img


__all__ = ["Layer", "to_rgb"]
