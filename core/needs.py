"""Turn a user's plain-language meal note into a small, explicit input patch.

The parser is intentionally online-only.  Without an explicitly configured
service the caller gets a Chinese hand-entry message rather than a guessed
local result.  The model output is treated as untrusted data and is narrowed
to fields that the original note actually mentions.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .llm_config import LLMConfigError, create_client, get_llm_config


MAX_NEEDS_LENGTH = 4000
_FIELDS = {
    "meal_budget", "cooking_willingness", "shopping_willingness", "fridge",
    "taste", "special_needs", "hunger", "time", "steps", "screen_hours",
    "recent_diet", "wish",
}
_ENUMS = {
    "cooking_willingness": {"愿意", "看情况", "不愿意"},
    "shopping_willingness": {"愿意", "看情况", "不愿意"},
    "special_needs": {"无", "减脂", "增肌", "素食"},
}


class NeedsError(RuntimeError):
    """A safe, user-facing plain-language parsing error."""


def _mentions(field: str, text: str) -> bool:
    chinese_number = r"[零一二两三四五六七八九十百千万]+"
    numeric = rf"(?:\d+(?:\.\d+)?|{chinese_number})"
    patterns = {
        "meal_budget": rf"(?:预算|花费|最多花|消费)\s*(?:是|为|大约|约|还剩)?\s*(?:人民币|¥|￥)?\s*{numeric}\s*(?:元|块)?",
        "cooking_willingness": r"做饭|下厨|开火|烹饪|自己做",
        "shopping_willingness": r"买菜|买食材|采购|补食材",
        "fridge": r"冰箱|库存|现有食材|家里有|家中有|我有(?:鸡蛋|番茄|蔬菜|食材)",
        "taste": r"口味|喜欢|想吃|偏好|辣|清淡|酸|甜|咸|麻|香",
        "special_needs": r"减脂|增肌|素食|无特殊|没有特殊",
        "hunger": rf"(?:饿|饥饿|饥饿度|饿度)\s*(?:是|到|为)?\s*{numeric}\s*(?:分|级)?",
        "time": rf"(?:\d{{1,2}}\s*[:：点]\s*\d{{0,2}}\s*分?|{chinese_number}点(?:{chinese_number}分?)?)",
        "steps": rf"(?:步数|走了)\s*(?:是|有)?\s*{numeric}\s*步",
        "screen_hours": rf"(?:屏幕|手机|电脑)[^。！？,，]*?{numeric}\s*小时|{numeric}\s*小时[^。！？,，]*(?:屏幕|手机|电脑)",
        "recent_diet": r"最近|近几天|近3天|饮食|吃了什么|连续外卖",
        "wish": r"想|希望|最好|要求|需要|尽量|帮我|不想",
    }
    return bool(re.search(patterns[field], text, flags=re.I))


def _prompt(text: str) -> str:
    schema = {
        "meal_budget": "非负数字",
        "cooking_willingness": "愿意/看情况/不愿意",
        "shopping_willingness": "愿意/看情况/不愿意",
        "fridge": "明确提到的冰箱食材原文",
        "taste": "明确口味",
        "special_needs": "无/减脂/增肌/素食",
        "hunger": "1到10的整数",
        "time": "HH:MM",
        "steps": "非负整数",
        "screen_hours": "非负数字",
        "recent_diet": "明确提到的近期饮食",
        "wish": "未能归入上述字段的明确要求原文",
    }
    return (
        "请把用户这段话整理成一个 JSON 对象。只填写用户明确提及的字段；未提及的字段必须省略，不能根据常识补默认值或猜测。"
        "复杂、无法归入固定字段的要求原文放进 wish；不要把模型自己的建议写入任何字段。"
        f"允许字段及格式：{json.dumps(schema, ensure_ascii=False)}。"
        "只返回 JSON 对象，不要 Markdown、代码围栏或解释。\n用户原话：" + text
    )


def _as_number(value: Any, field: str) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if not match:
            return None
        value = match.group(0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    if field in {"hunger", "steps"}:
        if not number.is_integer():
            return None
        return int(number)
    return number


def _as_time(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().replace("点", ":").replace("时", ":")
    value = re.sub(r"分$", "", value)
    match = re.fullmatch(r"(\d{1,2}):(\d{1,2})", value)
    if not match:
        return None
    hour, minute = map(int, match.groups())
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def _validate_patch(raw: Any, text: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise NeedsError("需求整理结果不是可用的内容，请改为手动填写")
    patch: dict[str, Any] = {}
    for field, value in raw.items():
        if field not in _FIELDS or not _mentions(field, text):
            continue
        if field in _ENUMS:
            if isinstance(value, str) and value in _ENUMS[field]:
                patch[field] = value
            continue
        if field in {"meal_budget", "screen_hours"}:
            number = _as_number(value, field)
            if number is not None:
                patch[field] = float(number)
            continue
        if field in {"hunger", "steps"}:
            number = _as_number(value, field)
            if number is not None and ((field == "hunger" and 1 <= number <= 10) or (field == "steps" and number >= 0)):
                patch[field] = number
            continue
        if field == "time":
            parsed = _as_time(value)
            if parsed is not None:
                patch[field] = parsed
            continue
        if isinstance(value, str) and value.strip():
            patch[field] = value.strip()
    return patch


def parse_needs(text: str) -> dict[str, Any]:
    """Parse only explicit needs from *text* using the configured service."""
    if not isinstance(text, str) or not text.strip():
        raise NeedsError("请先写下今晚的用餐想法")
    if len(text) > MAX_NEEDS_LENGTH:
        raise NeedsError("这段话太长了，请控制在4000字以内")
    config = get_llm_config()
    if config.parliament_mode in {"mock", "demo"} or not config.configured:
        raise NeedsError("当前没有在线整理服务，请手动填写下面的用餐需求")
    try:
        client = create_client(config)
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "你是谨慎的用餐需求整理助手，只输出 JSON。"},
                {"role": "user", "content": _prompt(text.strip())},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=60,
            **config.request_options,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise NeedsError("需求整理没有返回内容，请改为手动填写")
        raw = json.loads(content)
        return _validate_patch(raw, text.strip())
    except NeedsError:
        raise
    except (LLMConfigError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise NeedsError("需求整理没有完成，请检查文字后重试或手动填写") from None
    except Exception:
        raise NeedsError("在线整理暂时没有完成，请稍后重试或手动填写") from None


__all__ = ["NeedsError", "MAX_NEEDS_LENGTH", "parse_needs"]
