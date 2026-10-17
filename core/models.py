"""Typed data exchanged by the meal parliament.

The app deliberately keeps the trace as ordinary JSON-compatible models. That
makes the demo auditable and lets a user download exactly what was displayed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserInput(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    time: str = Field(default="19:00", description="当地时间 HH:MM")
    hunger: int = Field(default=5, ge=1, le=10)
    steps: int = Field(default=5000, ge=0)
    screen_hours: float = Field(default=8, ge=0)
    weekly_spend: float = Field(default=200, ge=0)
    budget: float = Field(default=800, ge=0)
    meal_budget: float | None = Field(default=None, ge=0)
    fridge: str = ""
    recent_diet: str = ""
    taste: str = ""
    special_needs: str = "无"
    wish: str = ""
    cooking_willingness: Literal["愿意", "看情况", "不愿意"] = "看情况"
    shopping_willingness: Literal["愿意", "看情况", "不愿意"] = "看情况"

    @field_validator("time")
    @classmethod
    def valid_time(cls, value: str) -> str:
        value = value.strip()
        try:
            hour, minute = value.split(":", 1)
            if not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("时间应为 HH:MM，例如 21:30") from None
        return f"{int(hour):02d}:{int(minute):02d}"

    @property
    def hour(self) -> float:
        hour, minute = self.time.split(":")
        return int(hour) + int(minute) / 60

    @property
    def remaining_budget(self) -> float:
        """Money available for this meal, with legacy weekly fields as fallback."""
        if self.meal_budget is not None:
            return self.meal_budget
        return max(0.0, self.budget - self.weekly_spend)

    @property
    def vegetarian(self) -> bool:
        return "素食" in self.special_needs

    def prompt_dict(self) -> dict[str, Any]:
        """Public prompt fields; legacy weekly fields stay internal when meal budget is set."""
        data = self.model_dump()
        if self.meal_budget is not None:
            data.pop("weekly_spend", None)
            data.pop("budget", None)
        return data

    def prompt_json(self) -> str:
        import json
        return json.dumps(self.prompt_dict(), ensure_ascii=False)

    @property
    def fingerprint(self) -> str:
        import hashlib
        import json

        raw = json.dumps(self.model_dump(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class Message(BaseModel):
    id: str
    role: str
    emoji: str
    name: str
    text: str
    weight: float
    round: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: bool = False


class Proposal(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    id: str
    mode: Literal["做饭", "外卖"]
    dish: str
    reason: str
    estimated_cost: float = Field(ge=0)
    nutrition: str
    steps: list[str] = Field(min_length=1)
    ingredients: list[str] = Field(default_factory=list)
    missing_ingredients: list[str] = Field(default_factory=list)
    feasible: bool = True
    constraint_note: str = ""


class Vote(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    agent: str
    emoji: str
    round: int
    scores: dict[str, float]
    rationale: str
    cited_message_id: str | None = None
    cited_excerpt: str | None = None


class Recommendation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    mode: Literal["做饭", "外卖"]
    dish: str
    reason: str
    estimated_cost: float
    nutrition: str
    steps: list[str]
    disagreements: str


class ParliamentResult(BaseModel):
    input_fingerprint: str
    weights: dict[str, dict[str, Any]]
    messages: list[Message]
    summary: str
    adjustments: list[dict[str, Any]]
    proposals: list[Proposal]
    votes: list[Vote]
    weighted_scores: dict[str, float]
    normalized_scores: dict[str, float]
    winning_proposal_id: str
    extra_round: bool
    recommendation: Recommendation
    trace: list[dict[str, Any]]

    def export_dict(self) -> dict[str, Any]:
        """Export the user-facing recommendation contract plus the full trace."""
        return {**self.recommendation_dict(), **self.trace_dict()}

    def recommendation_dict(self) -> dict[str, Any]:
        rec = self.recommendation
        return {
            "recommendation": rec.mode,
            "dish": rec.dish,
            "reason": rec.reason,
            "estimated_cost": rec.estimated_cost,
            "nutrition": rec.nutrition,
            "steps": rec.steps,
            "disagreements": rec.disagreements,
        }

    def trace_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace,
            "input_fingerprint": self.input_fingerprint,
            "weights": self.weights,
            "messages": [message.model_dump(mode="json") for message in self.messages],
            "summary": self.summary,
            "adjustments": self.adjustments,
            "proposals": [proposal.model_dump(mode="json") for proposal in self.proposals],
            "votes": [vote.model_dump(mode="json") for vote in self.votes],
            "weighted_scores": self.weighted_scores,
            "normalized_scores": self.normalized_scores,
            "winning_proposal_id": self.winning_proposal_id,
            "extra_round": self.extra_round,
        }
