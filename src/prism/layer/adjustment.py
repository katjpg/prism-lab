from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prism.color.adjust import Adjustment


@dataclass
class AdjustmentLayer:
    """Applies a sequence of adjustments to an RGB image.

    In a canvas stack, the image is whatever has been composited below.

    Parameters
    ----------
    steps : tuple[Adjustment, ...]
        Adjustments to apply, in order.
    """

    steps: tuple[Adjustment, ...]

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Return ``rgb`` with each adjustment applied in order."""
        out = rgb
        for step in self.steps:
            out = step.apply(out)
        return out


__all__ = ["AdjustmentLayer"]
