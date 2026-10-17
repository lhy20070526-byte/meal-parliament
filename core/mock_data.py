"""Input-driven mock content used when the selected provider is unconfigured."""

from __future__ import annotations

from .models import Proposal, UserInput


def has_token(text: str, *tokens: str) -> bool:
    text = text.lower()
    return any(token.lower() in text for token in tokens)


def has_all(text: str, *tokens: str) -> bool:
    """Require every named ingredient before calling a recipe stocked."""
    lowered = text.lower()
    return all(token.lower() in lowered for token in tokens)


def missing_items(text: str, items: tuple[str, ...]) -> list[str]:
    return [item for item in items if item.lower() not in text.lower()]


def proposal_seed(user: UserInput) -> list[Proposal]:
    fridge = user.fridge
    empty = not fridge.strip() or has_token(fridge, "空", "没有")
    vegetarian = user.vegetarian
    home_feasible = True
    home_note = ""
    home_missing: list[str] = []
    home_steps: list[str]
    if vegetarian:
        needed = ("豆腐", "西兰花", "糙米")
        missing = missing_items(fridge, needed)
        home_dish = "豆腐西兰花糙米碗"
        if missing:
            home_feasible = False
            home_cost = 25.0
            home_note = "缺少：" + "、".join(missing)
            home_missing = missing
            home_reason = f"按纯素保守处理，但库存缺少{ '、'.join(missing) }；禁用做饭候选，采购估算约¥25"
        else:
            home_cost = 0.0
            home_reason = "按纯素保守处理，三样主食材均在库存；估算新增成本¥0，调味缺口另计"
        home_steps = ["确认豆腐、西兰花、糙米均在保质期", "豆腐煎热、西兰花焯水", "糙米加热装盘；全程约15分钟"]
    elif has_all(fridge, "鸡胸", "西兰花", "糙米"):
        home_dish, home_reason, home_cost = "鸡胸肉西兰花糙米", "鸡胸、西兰花、糙米均已核验在库存；估算新增成本¥0，调味缺口另计", 0.0
        home_steps = ["确认鸡胸、西兰花、糙米均在保质期", "鸡胸切片煎熟，西兰花焯水", "糙米加热装盘；全程约20分钟"]
    elif has_all(fridge, "鸡蛋", "白菜", "剩饭"):
        unknown_storage = any(token in fridge for token in ("储存未知", "储存不明", "存储未知", "存储不明", "保存未知", "保存不明", "存放未知", "存放不明"))
        if unknown_storage:
            home_dish = "不设为家中剩饭候选"
            home_reason = "剩饭储存未知，不以气味判断可食，也不把它作为今晚候选；避免假装已有安全食材"
            home_cost = 0.0
            home_feasible = False
            home_note = "剩饭储存未知，不作为候选食材"
            home_steps = ["不使用储存未知的剩饭", "改评估可核验食材或外卖"]
        else:
            home_dish, home_reason, home_cost = "鸡蛋白菜拌饭", "鸡蛋、白菜和剩饭均已列出；估算新增成本¥0，使用前需按实际储存记录核实安全", 0.0
            home_steps = ["按实际储存记录确认剩饭安全", "白菜切丝、鸡蛋炒熟", "与剩饭拌匀；全程约12分钟"]
    elif empty:
        home_dish, home_reason, home_cost = "不设为家中现做候选", "冰箱为空，避免假装已有食材", 0.0
        home_feasible = False
        home_note = "冰箱为空"
        home_missing = ["鸡蛋", "蔬菜", "主食", "食用油", "盐"]
        home_steps = ["补充可用食材后再评估做饭"]
    else:
        if any(token in fridge for token in ("鸡胸", "西兰花", "糙米")):
            missing = missing_items(fridge, ("鸡胸肉", "西兰花", "糙米"))
            home_missing = missing
            home_reason = f"库存不足以完成鸡胸肉西兰花糙米，缺少：{'、'.join(missing)}；禁用做饭候选，采购估算约¥20"
        elif any(token in fridge for token in ("鸡蛋", "白菜", "剩饭")):
            missing = missing_items(fridge, ("鸡蛋", "白菜", "剩饭"))
            home_missing = missing
            home_reason = f"库存不足以完成鸡蛋白菜拌饭，缺少：{'、'.join(missing)}；禁用做饭候选，采购估算约¥20"
        else:
            home_reason = "当前库存没有匹配的完整演示菜谱；避免把缺项说成已有"
        home_dish, home_cost = "不设为家中现做候选", 20.0
        home_feasible = False
        home_note = "缺少完整菜谱所需食材"
        home_steps = ["补充完整食材后再评估做饭"]

    if vegetarian:
        takeout_dish, takeout_cost = "纯素豆腐蔬菜套餐", 32.0
    elif has_token(user.wish, "日料") or has_token(user.taste, "日料"):
        takeout_dish, takeout_cost = "日式低油定食", 48.0
    elif has_token(user.taste, "辣", "重口"):
        takeout_dish, takeout_cost = "麻辣烫（多蔬菜版）", 35.0
    else:
        takeout_dish, takeout_cost = "低油高蛋白蔬菜套餐", 32.0

    delivery_steps = ["选择可见营养信息的商家", "备注纯素、少油少盐", "收到后先确认分量与过期风险"] if vegetarian else ["选择能显示配料和配送费的商家", "备注少油少盐并确认主食/蛋白质", "收到后先核对分量再停止加点"]
    delivery = Proposal(
        id="P2", mode="外卖", dish=takeout_dish,
        reason="把口味与执行成本纳入选择；估算为一次外卖总额（含基础配送假设），下单前核对配料和配送费；素食时核对纯素标识",
        estimated_cost=takeout_cost, nutrition="建议少油少盐，补一份蔬菜和明确蛋白质",
        steps=delivery_steps,
        ingredients=[],
    )
    quick_steps = ["确认商品配料带纯素标识", "按饥饿程度分份", "补水并停止继续加点"] if vegetarian else ["确认商品配料、保质和配送费", "按饥饿程度分份", "补水并停止继续加点"]
    quick = Proposal(
        id="P3", mode="外卖",
        dish="便利店纯素蛋白质+蔬菜组合" if vegetarian else "便利店蛋白质+蔬菜组合",
        reason="把准备时间压到最低，估算为本次新增采购/购买总额，适合屏幕时间长或很饿的时段",
        estimated_cost=28.0,
        nutrition="蛋白质、碳水和蔬菜尽量齐全，估算不替代医学建议",
        steps=quick_steps,
        ingredients=[],
    )
    home = Proposal(
        id="P1", mode="做饭", dish=home_dish, reason=home_reason,
        estimated_cost=home_cost, nutrition="按食材可得性估算，营养信息为非精确建议",
        steps=home_steps,
        ingredients=[x for x in ("鸡蛋", "白菜", "剩饭", "鸡胸肉", "西兰花", "糙米", "豆腐", "牛奶") if x in home_dish],
        missing_ingredients=home_missing,
        feasible=home_feasible,
        constraint_note=home_note,
    )
    return [home, delivery, quick]
