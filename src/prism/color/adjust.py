from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from prism.color.palette import RECOLOR_SCHEMES, recolor
from prism.color.value import (
    Value,
    ValueMethod,
    bilateral_smooth,
    flatten_illumination,
)


@dataclass(frozen=True, slots=True)
class ColorWheel:
    """Hue, saturation, and lightness adjustment.

    Parameters
    ----------
    hue : float
        Degrees to rotate the hue.
    saturation : float
        Saturation multiplier. ``1.0`` leaves saturation unchanged.
    lightness : float
        Lightness shift in ``[-1, 1]``. Positive (+) values lighten; negative (-)
        values darken.
    """

    hue: float = 0.0
    saturation: float = 1.0
    lightness: float = 0.0


def apply_color_wheel(rgb: np.ndarray, color: ColorWheel) -> np.ndarray:
    """Apply a hue, saturation, and lightness adjustment to ``rgb``.

    Parameters
    ----------
    rgb : np.ndarray
        Image of shape ``(H, W, 3)``, float32, range ``0..1``.
    color : ColorWheel
        Adjustment to apply.

    Returns
    -------
    np.ndarray
        Adjusted image, shape ``(H, W, 3)``, float32, range ``0..1``.
    """
    out = np.clip(rgb.astype("float32"), 0.0, 1.0)

    if abs(color.hue) > 1e-8:
        out = hue_rotate(out, color.hue)

    if abs(color.saturation - 1.0) > 1e-8:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * float(color.saturation), 0.0, 1.0)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if abs(color.lightness) > 1e-8:
        if color.lightness >= 0:
            out = out + (1.0 - out) * float(color.lightness)
        else:
            out = out * (1.0 + float(color.lightness))

    return np.clip(out, 0.0, 1.0).astype("float32")


def hue_rotate(rgb: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate the hue of ``rgb`` by ``degrees``.

    Parameters
    ----------
    rgb : np.ndarray
        Image of shape ``(H, W, 3)``, float32, range ``0..1``.
    degrees : float
        Degrees to rotate the hue.

    Returns
    -------
    np.ndarray
        Hue-rotated image, shape ``(H, W, 3)``, float32, range ``0..1``.
    """
    hsv = cv2.cvtColor(rgb.astype("float32"), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + float(degrees)) % 360.0

    return np.clip(
        cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB),
        0.0,
        1.0,
    ).astype("float32")


class Adjustment(Protocol):
    """Protocol for RGB image adjustments."""

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Return ``rgb`` with the adjustment applied."""
        ...


@dataclass(frozen=True, slots=True)
class Smooth:
    """Smooth an image while preserving edges.

    Parameters
    ----------
    sigma_color : float
        How different two pixels can be in color and still blend.
    sigma_space : float
        Neighborhood radius, in pixels.
    """

    sigma_color: float = 0.10
    sigma_space: float = 4.0

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Return a smoothed version of ``rgb``."""
        return bilateral_smooth(rgb, self.sigma_color, self.sigma_space)


@dataclass(frozen=True, slots=True)
class FlatLight:
    """Reduce broad lighting variation across an image.

    Parameters
    ----------
    strength : float
        Amount of low-frequency illumination to remove, in ``0..1``.
    """

    strength: float = 0.50

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Return ``rgb`` with broad lighting variation reduced."""
        return flatten_illumination(rgb, strength=self.strength)


@dataclass(frozen=True, slots=True)
class Quantize:
    """Reduce an image to a small number of value bands.

    Parameters
    ----------
    bands : int
        Number of value bands.
    method : {'multiotsu', 'kmeans', 'percentile'}
        Method used to split the tonal range.
    seed : int
        Seed for the stochastic ``kmeans`` method.
    """

    bands: int = 5
    method: ValueMethod = "multiotsu"
    seed: int = 7

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Return ``rgb`` reduced to ``bands`` value regions."""
        result = Value(n_bands=self.bands, method=self.method, seed=self.seed).extract(rgb)
        out = np.zeros_like(rgb, dtype="float32")
        for band in range(result.n_bands):
            mask = result.labels == band
            if mask.any():
                out[mask] = rgb[mask].mean(axis=0)
        return out


@dataclass(frozen=True, slots=True)
class ColorShift:
    """Shift hue, saturation, and lightness.

    Parameters
    ----------
    hue : float
        Degrees to rotate the hue.
    saturation : float
        Saturation multiplier. ``1.0`` leaves saturation unchanged.
    lightness : float
        Lightness shift in ``[-1, 1]``.
    """

    hue: float = 0.0
    saturation: float = 1.0
    lightness: float = 0.0

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Return ``rgb`` with the color shift applied."""
        return apply_color_wheel(
            rgb,
            ColorWheel(self.hue, self.saturation, self.lightness),
        )


@dataclass(frozen=True, slots=True)
class Recolor:
    """Remap image tones to a built-in color scheme.

    Parameters
    ----------
    scheme : str
        Name of a built-in recolor scheme (for example ``"sepia"`` or
        ``"teal_orange"``).
    strength : float
        Blend amount toward the scheme, in ``0..1``.
    vibrance : float
        Chroma multiplier for the recolored output.
    smoothness : float
        Spatial smoothing applied to the tonal mapping.
    seed : int
        Seed for value-band extraction.
    """

    scheme: str = "sepia"
    strength: float = 0.40
    vibrance: float = 0.85
    smoothness: float = 6.0
    seed: int = 7

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Return ``rgb`` recolored toward ``scheme``.

        Raises
        ------
        ValueError
            If ``scheme`` is not a known recolor scheme.
        """
        try:
            anchors = RECOLOR_SCHEMES[self.scheme]
        except KeyError as exc:
            raise ValueError(f"unknown recolor scheme: {self.scheme!r}") from exc

        value = Value(n_bands=5, smooth=True, seed=self.seed).extract(rgb)
        return recolor(
            rgb,
            value=value,
            anchors=anchors,
            amount=self.strength,
            chroma=self.vibrance,
            sigma=self.smoothness,
        )


__all__ = [
    "Adjustment",
    "ColorShift",
    "ColorWheel",
    "FlatLight",
    "Quantize",
    "Recolor",
    "Smooth",
    "apply_color_wheel",
    "hue_rotate",
]
