"""Deterministic starting weights for the five specialists."""

from __future__ import annotations

import re
from typing import Any

from .models import UserInput


AGENT_LABELS = {
    "营养师": "🥗",
    "钱包": "💰",
    "懒惰": "🛋️",
    "冰箱": "🧊",
    "口味": "🌶️",
}


def _expired_or_imminent(fridge: str) -> tuple[bool, bool]:
    text = fridge.lower().replace(" ", "")
    imminent = "临期" in text or bool(re.search(r"\d+天(?:后)?过期", text))
    # “3天过期” is interpreted as “expires in three days”, per the demo rule.
    # Remove future-expiry phrases first so a fridge containing both
    # “3天后过期” and “已过期” keeps both signals.
    remainder = re.sub(r"\d+天(?:后)?过期", "", text)
    expired = bool(re.search(r"(?:已|已经|今天|昨日|昨天)?过期(?:了|食品)?", remainder))
    return expired, imminent


def calculate_weights(user_input: UserInput) -> dict[str, dict[str, Any]]:
    """Return auditable role weights and the rule evidence behind each value."""

    values = {name: 1.0 for name in AGENT_LABELS}
    reasons: dict[str, list[str]] = {name: ["基础权重 1.0"] for name in AGENT_LABELS}

    if user_input.hunger >= 8 or user_input.hour >= 21:
        values["懒惰"] *= 2
        reasons["懒惰"].append("饥饿度≥8或时间≥21:00，偏好低摩擦方案 ×2")
        values["钱包"] *= 0.7
        reasons["钱包"].append("饥饿度≥8或时间≥21:00，降低等待/价格敏感度 ×0.7")

    # A per-meal budget supersedes the legacy weekly fields.  Keep the old
    # rule only for callers that still construct the legacy input shape.
    if user_input.meal_budget is None and user_input.weekly_spend > user_input.budget * 0.8:
        values["钱包"] *= 2
        reasons["钱包"].append("本周支出严格超过预算的80%，预算约束 ×2")

    expired, imminent = _expired_or_imminent(user_input.fridge)
    if expired or imminent:
        values["冰箱"] *= 1.5
        if expired:
            reasons["冰箱"].append("冰箱包含已过期描述，安全检查 ×1.5")
        else:
            reasons["冰箱"].append("冰箱包含临期/即将过期食材，利用窗口 ×1.5")

    diet = f"{user_input.recent_diet}{user_input.special_needs}".lower()
    if any(token in diet for token in ("蔬菜少", "蔬菜不足", "少蔬菜", "没有蔬菜", "没吃蔬菜", "缺乏蔬菜", "连续外卖")):
        values["营养师"] *= 1.5
        reasons["营养师"].append("近3天饮食显示蔬菜不足，营养补足 ×1.5")

    if user_input.steps > 12000:
        values["懒惰"] *= 1.3
        reasons["懒惰"].append("步数>12000，恢复成本权重 ×1.3")

    if user_input.cooking_willingness == "愿意":
        values["冰箱"] *= 1.15
        reasons["冰箱"].append("用户愿意做饭，库存方案偏好 ×1.15")
    elif user_input.cooking_willingness == "不愿意":
        values["懒惰"] *= 1.2
        reasons["懒惰"].append("用户不愿意做饭，降低执行摩擦 ×1.2")
    if user_input.shopping_willingness == "愿意":
        values["冰箱"] *= 1.1
        reasons["冰箱"].append("用户愿意买菜，补齐缺料方案偏好 ×1.1")
    elif user_input.shopping_willingness == "不愿意":
        values["冰箱"] *= 1.2
        reasons["冰箱"].append("用户不愿意买菜，严格核对现有库存 ×1.2")

    if "今天想吃顿好的" in user_input.wish:
        values["钱包"] *= 0.5
        reasons["钱包"].append("一句话需求想吃顿好的，价格敏感度 ×0.5")
        values["口味"] *= 1.5
        reasons["口味"].append("一句话需求想吃顿好的，体验偏好 ×1.5")

    if "减脂" in user_input.special_needs:
        values["营养师"] *= 1.8
        reasons["营养师"].append("特殊需求为减脂，营养约束 ×1.8")
        values["口味"] *= 0.8
        reasons["口味"].append("特殊需求为减脂，重口偏好需收敛 ×0.8")
    if "增肌" in user_input.special_needs:
        values["营养师"] *= 1.5
        reasons["营养师"].append("特殊需求为增肌，蛋白质约束 ×1.5")
        values["钱包"] *= 0.8
        reasons["钱包"].append("特殊需求为增肌，成本可让步 ×0.8")

    return {
        name: {"emoji": AGENT_LABELS[name], "value": round(value, 4), "reasons": evidence}
        for name, value, evidence in ((n, values[n], reasons[n]) for n in AGENT_LABELS)
    }
