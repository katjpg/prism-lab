from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from prism.paint.ground import Background, create_canvas


@dataclass(frozen=True, slots=True)
class Ground:
    """The opaque base that canvas layers composite over.

    Parameters
    ----------
    background : Background
        Fill style: a solid tone, the source image, or a paper texture.
    """

    background: Background = "white"

    def base(self, like: np.ndarray, source: np.ndarray | None = None) -> np.ndarray:
        """Return a ground image shaped as ``like``.

        Parameters
        ----------
        like : np.ndarray
            Array whose ``(H, W)`` the ground matches.
        source : np.ndarray or None, optional
            Reference image, needed for the ``mean``/``source``/texture
            grounds; resized to match when its shape differs.
        """
        if source is None or self.background in ("white", "black"):
            return create_canvas(like, self.background)
        h, w = like.shape[:2]
        if source.shape[:2] != (h, w):
            source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
        return create_canvas(source, self.background)


__all__ = ["Background", "Ground"]
