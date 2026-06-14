from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal, Protocol

from prism.color.value import ValueMethod
from prism.raster.geometric import ShapeFamily

RasterFill = Literal["value", "color"]
BoundaryFill = Literal["value", "color", "boundaries"]
RegionMethod = Literal["slic", "watershed"]


class RasterStyle(Protocol):
    category: ClassVar[str]
    mode: ClassVar[str]

    def params(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Stipple:
    category: ClassVar[str] = "abstract"
    mode: ClassVar[str] = "stipple"

    dots: int = 8000
    dot_size: float = 0.75
    darkness: float = 1.4
    edges: float = 0.35
    smoothness: float = 0.6
    paper_white: float = 0.08

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Geometric:
    category: ClassVar[str] = "abstract"
    mode: ClassVar[str] = "geometric"

    shapes: int = 120
    shape_type: ShapeFamily = "combo"
    opacity: float = 0.6
    edges: float = 0.5
    search: int = 8
    image_fit: float = 1.0
    effort: float = 0.45

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Boundary:
    category: ClassVar[str] = "abstract"
    mode: ClassVar[str] = "segment"

    regions: int = 600
    edge_fit: float = 10.0
    fill: BoundaryFill = "value"
    method: RegionMethod = "slic"
    smoothness: float = 1.0
    markers: int = 250
    watershed_tightness: float = 0.001

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValueScale:
    category: ClassVar[str] = "abstract"
    mode: ClassVar[str] = "posterize"

    levels: int = 5
    fill: RasterFill = "value"
    smoothness: float = 0.25
    method: ValueMethod = "multiotsu"
    perceived_value: bool = True
    even_lighting: bool = False

    def params(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "Boundary",
    "BoundaryFill",
    "Geometric",
    "RasterFill",
    "RasterStyle",
    "RegionMethod",
    "Stipple",
    "ValueScale",
]
