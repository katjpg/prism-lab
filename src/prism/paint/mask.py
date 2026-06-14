from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from prism.paint.brush import (
    DEFAULT_SEED,
    create_brush,
    rasterize_stroke,
    resample_path,
)


EPS = 1e-6

DEFAULT_STAMP_ANGLES = 36
DEFAULT_LEN_STEP = 4
DEFAULT_WID_STEP = 2

TEMPLATE_LENGTH = 200
TEMPLATE_WIDTH = 44
BRISTLES_PER_RADIUS = 7.0
BRISTLES_MIN = 8
BRISTLES_MAX = 120


MaskBuilder = Callable[[float, float, float], tuple[np.ndarray, float, float]]


def soft_stroke_mask(
    length: float,
    width: float,
    angle_deg: float,
    aa: float = 2.0,
) -> tuple[np.ndarray, float, float]:
    """Build a soft-edged stroke mask.

    Parameters
    ----------
    length : float
        Stroke length, in pixels.
    width : float
        Stroke width, in pixels.
    angle_deg : float
        Stroke angle, in degrees.
    aa : float, default=2.0
        Width of the antialiased edge transition, in pixels.

    Returns
    -------
    mask : np.ndarray
        Stroke coverage mask, shape ``(H, W)``, float32, range ``0..1``.
    cy : float
        Row coordinate of the mask center.
    cx : float
        Column coordinate of the mask center.

    Notes
    -----
    The mask is computed from the distance to a rotated line segment with a
    soft boundary ramp.
    """
    r = max(float(width) * 0.5, 0.5)
    half = float(length) * 0.5
    rad = np.deg2rad(angle_deg)
    bw = int(np.ceil(length * abs(np.cos(rad)) + width * abs(np.sin(rad)))) + int(2 * aa) + 3
    bh = int(np.ceil(length * abs(np.sin(rad)) + width * abs(np.cos(rad)))) + int(2 * aa) + 3
    cy, cx = bh / 2.0, bw / 2.0
    ys, xs = np.mgrid[0:bh, 0:bw].astype(np.float32)
    ax, ay = cx - half * np.cos(rad), cy - half * np.sin(rad)
    bx, by = cx + half * np.cos(rad), cy + half * np.sin(rad)
    abx, aby = bx - ax, by - ay
    apx, apy = xs - ax, ys - ay

    # project onto segment
    denom = abx * abx + aby * aby + 1e-9
    t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    dxp = apx - t * abx
    dyp = apy - t * aby
    dist = np.sqrt(dxp * dxp + dyp * dyp)

    # distance to coverage
    cov = np.clip(0.5 + (r - dist) / aa, 0.0, 1.0).astype(np.float32)
    return cov, cy, cx


def _rotate_to_bounds(
    img: np.ndarray,
    angle_deg: float,
    pad: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Rotate ``img`` into a bounding box that contains the result."""
    h, w = img.shape[:2]
    cX, cY = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += nW / 2.0 - cX
    M[1, 2] += nH / 2.0 - cY
    out = cv2.warpAffine(
        img,
        M,
        (max(1, nW), max(1, nH)),
        flags=cv2.INTER_LINEAR,
        borderValue=pad,
    )
    return out, out.shape[0] / 2.0, out.shape[1] / 2.0


class TemplateCache:
    """Cache stroke templates quantized by size and angle.

    Parameters
    ----------
    template_gray : np.ndarray
        Base template image in grayscale.
    n_angles : int, default=DEFAULT_STAMP_ANGLES
        Number of angle bins over ``[0, 180)``.
    len_step : int, default=DEFAULT_LEN_STEP
        Quantization step for stroke length, in pixels.
    wid_step : int, default=DEFAULT_WID_STEP
        Quantization step for stroke width, in pixels.

    Notes
    -----
    Requests are quantized before lookup so nearby stroke sizes and angles
    reuse the same cached template.
    """

    def __init__(
        self,
        template_gray: np.ndarray,
        n_angles: int = DEFAULT_STAMP_ANGLES,
        len_step: int = DEFAULT_LEN_STEP,
        wid_step: int = DEFAULT_WID_STEP,
    ) -> None:
        self.t = template_gray.astype(np.float32) / 255.0
        self.n_angles = int(n_angles)
        self.len_step = float(len_step)
        self.wid_step = float(wid_step)
        self._cache: dict[tuple[int, int, int], tuple[np.ndarray, float, float]] = {}

    def _key(self, length: float, width: float, angle: float) -> tuple[int, int, int]:
        """Quantize a stroke request into a cache key."""
        qa = int(round((angle % 180.0) / (180.0 / self.n_angles))) % self.n_angles
        ql = max(1, int(round(length / self.len_step)))
        qw = max(1, int(round(width / self.wid_step)))
        return ql, qw, qa

    def get(
        self,
        length: float,
        width: float,
        angle: float,
    ) -> tuple[np.ndarray, float, float]:
        """Return a cached template for ``length``, ``width``, and ``angle``.

        Parameters
        ----------
        length : float
            Stroke length, in pixels.
        width : float
            Stroke width, in pixels.
        angle : float
            Stroke angle, in degrees.

        Returns
        -------
        mask : np.ndarray
            Cached template mask, shape ``(H, W)``, float32, range ``0..1``.
        cy : float
            Row coordinate of the mask center.
        cx : float
            Column coordinate of the mask center.
        """
        k = self._key(length, width, angle)
        e = self._cache.get(k)
        if e is None:
            ql, qw, qa = k
            resized = cv2.resize(
                self.t,
                (max(1, int(ql * self.len_step)), max(1, int(qw * self.wid_step))),
                interpolation=cv2.INTER_AREA,
            )
            alpha, cy, cx = _rotate_to_bounds(
                resized,
                -qa * (180.0 / self.n_angles),
            )
            e = (np.ascontiguousarray(alpha, dtype=np.float32), cy, cx)
            self._cache[k] = e
        return e

    def __len__(self) -> int:
        """Return the number of cached templates."""
        return len(self._cache)


def _bristle_count(radius: float) -> int:
    """Map brush radius to a bounded bristle count."""
    return int(
        np.clip(
            round(radius * BRISTLES_PER_RADIUS),
            BRISTLES_MIN,
            BRISTLES_MAX,
        )
    )


def bristle_stroke_mask(
    length: float,
    width: float,
    angle_deg: float,
    seed: int = DEFAULT_SEED,
    soften: float = 0.35,
) -> tuple[np.ndarray, float, float]:
    """Rasterize a bristle stroke mask.

    Parameters
    ----------
    length : float
        Stroke length, in pixels.
    width : float
        Stroke width, in pixels.
    angle_deg : float
        Stroke angle, in degrees.
    seed : int, default=DEFAULT_SEED
        Seed used for brush generation and rasterization.
    soften : float, default=0.35
        Standard deviation of the Gaussian blur applied after rasterization.

    Returns
    -------
    mask : np.ndarray
        Stroke coverage mask, shape ``(H, W)``, float32, range ``0..1``.
    cy : float
        Row coordinate of the mask center.
    cx : float
        Column coordinate of the mask center.

    Notes
    -----
    The stroke is rendered by sampling a synthetic filbert brush along a
    straight path.
    """
    radius = max(width / 2.0, 1.0)
    brush = create_brush(
        radius=radius,
        n_bristles=_bristle_count(radius),
        shape="filbert",
        seed=seed,
    )
    rad = np.deg2rad(angle_deg)
    bh = int(np.ceil(length * abs(np.sin(rad)) + width * abs(np.cos(rad)))) + 16
    bw = int(np.ceil(length * abs(np.cos(rad)) + width * abs(np.sin(rad)))) + 16
    cy, cx = bh / 2.0, bw / 2.0
    half = length / 2.0
    p0 = np.array([cx - half * np.cos(rad), cy - half * np.sin(rad)], np.float32)
    p1 = np.array([cx + half * np.cos(rad), cy + half * np.sin(rad)], np.float32)

    # sample path
    path = resample_path(np.vstack([p0, p1]).astype(np.float32), step=1.5)

    # rasterize brush
    mask = rasterize_stroke(
        (bh, bw),
        path,
        brush,
        samples=10,
        grain_depth=0.18,
        grain_scale=0.12,
        render="line",
        seed=seed,
    )

    # soften mask
    if soften > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=soften)
    return mask.astype(np.float32), cy, cx


def render_bristle_template(
    length: int = TEMPLATE_LENGTH,
    width: int = TEMPLATE_WIDTH,
    seed: int = 0,
    soften: float = 0.5,
) -> np.ndarray:
    """Render a grayscale bristle template for caching.

    Parameters
    ----------
    length : int, default=TEMPLATE_LENGTH
        Template stroke length, in pixels.
    width : int, default=TEMPLATE_WIDTH
        Template stroke width, in pixels.
    seed : int, default=0
        Seed used for brush generation.
    soften : float, default=0.5
        Standard deviation of the Gaussian blur applied to the template.

    Returns
    -------
    np.ndarray
        Grayscale template image, shape ``(H, W)``, uint8, range ``0..255``.

    Notes
    -----
    The rendered stroke is tapered toward the template edges so rotated and
    resized variants remain visually smooth.
    """
    radius = width / 2.0
    brush = create_brush(
        radius=radius,
        n_bristles=_bristle_count(radius),
        shape="filbert",
        seed=seed,
    )
    bh, bw = width + 18, length + 18
    p0 = np.array([bw * 0.10, bh / 2.0], np.float32)
    p1 = np.array([bw * 0.90, bh / 2.0], np.float32)

    # sample path
    path = resample_path(np.vstack([p0, p1]).astype(np.float32), step=1.5)

    # rasterize brush
    mask = rasterize_stroke(
        (bh, bw),
        path,
        brush,
        samples=12,
        grain_depth=0.14,
        grain_scale=0.11,
        render="line",
        seed=7,
    )

    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    u = np.abs((xx - (bw - 1) / 2.0) / (bw / 2.0))
    v = np.abs((yy - (bh - 1) / 2.0) / (bh / 2.0))

    # taper along axes
    eu = np.clip((1.0 - u) / 0.30, 0.0, 1.0)
    eu = eu * eu * (3.0 - 2.0 * eu)
    ev = np.clip((1.0 - v) / 0.55, 0.0, 1.0)
    ev = ev * ev * (3.0 - 2.0 * ev)

    # apply taper
    out = mask * eu * ev

    # soften template
    if soften > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=soften)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def create_soft_mask_template(
    length: int = TEMPLATE_LENGTH,
    width: int = TEMPLATE_WIDTH,
    seed: int = 0,
    aa: float = 2.0,
    soften: float = 0.0,
) -> np.ndarray:
    """Render a grayscale soft-mask template for caching.

    Parameters
    ----------
    length : int, default=TEMPLATE_LENGTH
        Template stroke length, in pixels.
    width : int, default=TEMPLATE_WIDTH
        Template stroke width, in pixels.
    seed : int, default=0
        Seed used for the procedural streak texture.
    aa : float, default=2.0
        Width of the antialiased edge transition, in pixels.
    soften : float, default=0.0
        Standard deviation of the Gaussian blur applied to the template.

    Returns
    -------
    np.ndarray
        Grayscale template image, shape ``(H, W)``, uint8, range ``0..255``.

    Notes
    -----
    The template starts from an analytic soft stroke mask, then modulates it
    with a low-frequency streak texture.
    """
    cov, _, _ = soft_stroke_mask(length, width, 0.0, aa=aa)
    bh, bw = cov.shape
    rng = np.random.default_rng(seed)

    # generate streak field
    streak = rng.random((bh, bw)).astype(np.float32)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=18.0, sigmaY=1.2)

    # normalize texture
    streak = (streak - streak.min()) / (np.ptp(streak) + EPS)

    # modulate coverage
    tex = cov * (0.70 + 0.30 * streak)

    # soften template
    if soften > 0:
        tex = cv2.GaussianBlur(tex, (0, 0), sigmaX=float(soften))
    return (np.clip(tex, 0.0, 1.0) * 255.0).astype(np.uint8)


def create_bristle_cache() -> TemplateCache:
    """Create a cache initialized with a bristle template."""
    return TemplateCache(render_bristle_template())


def create_soft_cache() -> TemplateCache:
    """Create a cache initialized with a soft-mask template."""
    return TemplateCache(create_soft_mask_template())


_BRISTLE_CACHE: list[TemplateCache] = []
_SOFT_CACHE: list[TemplateCache] = []


def get_bristle_cache() -> TemplateCache:
    """Return the bristle template cache."""
    if not _BRISTLE_CACHE:
        _BRISTLE_CACHE.append(create_bristle_cache())
    return _BRISTLE_CACHE[0]


def get_soft_cache() -> TemplateCache:
    """Return the soft-mask template cache."""
    if not _SOFT_CACHE:
        _SOFT_CACHE.append(create_soft_cache())
    return _SOFT_CACHE[0]


__all__ = [
    "DEFAULT_LEN_STEP",
    "DEFAULT_STAMP_ANGLES",
    "DEFAULT_WID_STEP",
    "MaskBuilder",
    "TemplateCache",
    "bristle_stroke_mask",
    "soft_stroke_mask",
    "create_bristle_cache",
    "create_soft_cache",
    "create_soft_mask_template",
    "get_bristle_cache",
    "get_soft_cache",
    "render_bristle_template",
]
