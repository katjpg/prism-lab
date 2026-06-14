from __future__ import annotations

import io
from functools import cache, lru_cache
from importlib.resources import files
from typing import Literal

import cv2
import numpy as np
from PIL import Image

Background = Literal["white", "black", "mean", "source", "canvas", "sketchbook", "black-paper"]

BLACK = np.float32(0x20 / 255)
TEXTURES = ("canvas", "sketchbook", "black-paper")
PAINT_BORDER = 1.15
TEXTURE_BLEND = 0.01


@cache
def _texture(name: str) -> np.ndarray:
    data = (files("prism.paint") / "assets" / "textures" / f"{name}.jpg").read_bytes()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype="float32") / 255.0


@lru_cache(maxsize=8)
def _resized_texture(name: str, h: int, w: int) -> np.ndarray:
    tex = _texture(name)
    th, tw = tex.shape[:2]
    scale = max(h / th, w / tw)
    resized = cv2.resize(
        tex,
        (round(tw * scale), round(th * scale)),
        interpolation=cv2.INTER_AREA,
    )
    y = (resized.shape[0] - h) // 2
    x = (resized.shape[1] - w) // 2
    return np.ascontiguousarray(resized[y : y + h, x : x + w])


def load_texture(name: str, h: int, w: int) -> np.ndarray:
    """Return paper texture ``name`` cropped to ``(h, w)``."""
    return _resized_texture(name, h, w).copy()


def create_canvas(rgb: np.ndarray, background: str) -> np.ndarray:
    """Build a painting ground sized like ``rgb`` for the named background."""
    h, w = rgb.shape[:2]

    match background:
        case "source":
            return rgb.astype("float32").copy()
        case "white":
            return np.ones((h, w, 3), dtype="float32")
        case "black":
            return np.full((h, w, 3), BLACK, dtype="float32")
        case "mean":
            canvas = np.empty((h, w, 3), dtype="float32")
            canvas[:] = rgb.reshape(-1, 3).mean(axis=0)
            return canvas
        case _ if background in TEXTURES:
            return load_texture(background, h, w)
        case _:
            raise ValueError(f"unknown background: {background!r}")


def reflect_border(canvas: np.ndarray, pad_px: int) -> np.ndarray:
    """Pad ``canvas`` by ``pad_px`` on all sides with reflected pixels."""
    return cv2.copyMakeBorder(canvas, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_REFLECT_101)


def blend_texture(image: np.ndarray, background: str) -> np.ndarray:
    """Overlay a faint paper texture; a no-op for non-texture backgrounds."""
    if background not in TEXTURES:
        return image
    h, w = image.shape[:2]
    paper = load_texture(background, h, w)
    return np.clip(image * (1.0 - TEXTURE_BLEND) + paper * TEXTURE_BLEND, 0.0, 1.0).astype(
        "float32"
    )


__all__ = [
    "BLACK",
    "PAINT_BORDER",
    "TEXTURE_BLEND",
    "TEXTURES",
    "Background",
    "blend_texture",
    "create_canvas",
    "load_texture",
    "reflect_border",
]
