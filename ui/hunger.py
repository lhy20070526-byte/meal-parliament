"""Small local Streamlit component for an expressive hunger slider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_COMPONENT = components.declare_component(
    "meal_parliament_hunger_slider",
    path=str(Path(__file__).with_name("hunger_component")),
)


def _clamp(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 5
    return max(1, min(10, number))


def hunger_slider(value: int = 5, key: str | None = None) -> int:
    """Render a 1–10 slider and return its last committed value."""
    initial = _clamp(value)
    result = _COMPONENT(value=initial, default=initial, key=key)
    return _clamp(initial if result is None else result)


__all__ = ["hunger_slider"]
