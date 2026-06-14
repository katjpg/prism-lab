from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    """Tracks the active layer, tool, and control values during an edit.

    Parameters
    ----------
    active_layer : int or None
        Index of the selected layer in the canvas stack.
    active_tool : str or None
        Name of the selected tool.
    controls : dict[str, Any]
        Current control values for the active tool.
    """

    active_layer: int | None = None
    active_tool: str | None = None
    controls: dict[str, Any] = field(default_factory=dict)


__all__ = ["Session"]
