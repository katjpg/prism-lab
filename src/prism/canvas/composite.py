from __future__ import annotations

from typing import Literal

import numpy as np

BlendMode = Literal["normal", "multiply", "screen", "overlay", "soft_light"]


def composite(
    base: np.ndarray,
    over: np.ndarray,
    alpha: np.ndarray,
    *,
    mode: BlendMode = "normal",
    opacity: float = 1.0,
) -> np.ndarray:
    """Composite ``over`` onto ``base``.

    Parameters
    ----------
    base : np.ndarray
        Backdrop image, shape ``(H, W, 3)``, float32, range ``0..1``.
    over : np.ndarray
        Foreground image, shape ``(H, W, 3)``, float32, range ``0..1``.
    alpha : np.ndarray
        Foreground coverage, shape ``(H, W)``, float32, range ``0..1``.
    mode : {'normal', 'multiply', 'screen', 'overlay', 'soft_light'}
        Blend mode used where foreground and backdrop overlap.
    opacity : float
        Multiplier applied to ``alpha``, in ``0..1``.

    Returns
    -------
    np.ndarray
        Composited image, shape ``(H, W, 3)``, float32, range ``0..1``.

    Raises
    ------
    ValueError
        If ``mode`` is not a supported blend mode.
    """
    a = np.clip(alpha * float(opacity), 0.0, 1.0)[..., None]
    blended = _blend(base, over, mode)
    return base * (1.0 - a) + blended * a


def _blend(base: np.ndarray, over: np.ndarray, mode: BlendMode) -> np.ndarray:
    if mode == "normal":
        return over
    if mode == "multiply":
        return base * over
    if mode == "screen":
        return 1.0 - (1.0 - base) * (1.0 - over)
    if mode == "overlay":
        return np.where(
            base <= 0.5,
            2.0 * base * over,
            1.0 - 2.0 * (1.0 - base) * (1.0 - over),
        )
    if mode == "soft_light":
        return (1.0 - 2.0 * over) * base * base + 2.0 * over * base
    raise ValueError(f"unknown blend mode: {mode!r}")


__all__ = ["BlendMode", "composite"]
