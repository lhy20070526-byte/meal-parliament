"""Meal Parliament core orchestration and data models."""

from .models import UserInput, Proposal, Recommendation, ParliamentResult
from .weights import calculate_weights


def __getattr__(name):
    # Keep the public import without importing agents during models import.
    if name == "Parliament":
        from .parliament import Parliament
        return Parliament
    raise AttributeError(name)

__all__ = ["UserInput", "Proposal", "Recommendation", "ParliamentResult", "calculate_weights", "Parliament"]
