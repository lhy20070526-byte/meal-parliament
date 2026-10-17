"""The chair coordinates evidence; it never impersonates specialist votes."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

from core.mock_data import proposal_seed
from core.models import Message, Proposal, Recommendation, UserInput, Vote

from .base import BaseAgent, AgentError


def _explicitly_uses_leftover(text: str) -> bool:
    """Return whether a leftover mention is an affirmative use.

    Check each occurrence independently.  A nearby exclusion such as
    ``不使用牛奶`` must not hide a later ``加入剩饭`` in the same step, while
    labels such as ``无剩饭版`` and safety notes that discard the rice are
    ignored.
    """
    if not text:
        return False
    excluded_before = re.compile(
        r"(?:不使用|不用|不要使用|不得使用|不再使用|不碰|弃用|丢弃|扔掉|排除|"
        r"避免(?:使用)?|不添加|不加入|无)"
        r"(?:[\s、，,：:及和与]*(?:储存|存储|保存|存放)?(?:未知|不明)?[\s的]*)$"
    )
    excluded_after = re.compile(
        r"^[\s、，,：:]*"
        r"(?:[（(](?:储存|存储|保存|存放)?(?:未知|不明)[）)])?"
        r"(?:不使用|不用|不要使用|不得使用|不再使用|不碰|弃用|丢弃|扔掉|排除|"
        r"不纳入候选|不作为(?:候选|食材)|避免(?:使用)?|不添加|不加入)"
    )
    for match in re.finditer("剩饭", text):
        prefix = text[max(0, match.start() - 48):match.start()]
        suffix = text[match.end():match.end() + 32]
        if excluded_before.search(prefix) or excluded_after.search(suffix):
            continue
        return True
    return False


def _expiry_content(text: str) -> str:
    """Keep affirmative expiry mentions while removing narrow safety checks."""
    text = re.sub(r"\d+天(?:后)?过期", "", text)
    text = re.sub(r"尚未过期|未过期|没有过期|没过期|过期风险|检查是否过期|确认是否过期", "", text)
    return re.sub(r"(?:不使用|不用|不得使用|禁止使用|避免使用|不添加|不加入|丢弃|扔掉|排除)(?:任何|已|已经|明确)?过期", "", text)


def _valid_ingredient_name(value: Any) -> bool:
    """Reject blank, punctuation-only, and quantity-only ingredient entries."""
    if not isinstance(value, str):
        return False
    value = re.sub(r"[（(].*?[）)]", "", value).strip()
    if not value or not re.search(r"[A-Za-z\u4e00-\u9fff]", value):
        return False
    return not bool(re.fullmatch(r"[\d\s.,，。个只颗枚克千克g毫升ml升斤两份包袋]+", value, flags=re.I))


class Speaker(BaseAgent):
    name, emoji = "议长", "🎙️"
    display_name, job_title = "餐桌主持人", "用餐建议"
    system_prompt = "你是餐桌主持人，负责把五种真实冲突整理成今晚能执行的决定。请使用简体中文、人民币元和中国大陆常见家常做法；价格只做估计，不编造实时外卖价格。指出不同意见，不冒充五位顾问单独评分。安全边界：储存未知的剩饭不以气味判断可食，也不作为候选食材。"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_proposals: list[Proposal] = []
        self.proposal_attempts: list[list[dict[str, Any]]] = []
        self._proposal_payload: list[dict[str, Any]] | None = None
        self._repairing_proposals = False

    def _real_json(self, prompt: str) -> dict[str, Any]:
        try:
            prompt = (
                "请只返回一个合法的 JSON 对象（JSON object），不要返回 Markdown、代码围栏或额外文字。\n"
                + prompt
            )
            response = self._real_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
                **self._request_options(),
            )
            return json.loads((response.choices[0].message.content or "").strip())
        except Exception as exc:
            raise AgentError(f"议长真实模式JSON失败：{type(exc).__name__}，请检查密钥、JSON或重试") from exc

    def summarize(self, user_input: UserInput, messages: list[Message], weights: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        if not self.use_mock:
            data = self._real_json(
                "基于用户输入、专家发言和当前权重总结。只返回JSON："
                '{"summary":"...","adjustments":[{"agent":"钱包","delta":0.1,"reason":"证据"}]}。'
                f"\n用户：{user_input.prompt_json()}\n发言：{json.dumps([m.model_dump(mode='json') for m in messages], ensure_ascii=False)}\n参考：{json.dumps(weights, ensure_ascii=False)}"
            )
            adjustments = data.get("adjustments", [])
            allowed = set(weights)
            seen: set[str] = set()
            if not isinstance(adjustments, list):
                raise AgentError("议长微调必须是列表")
            for adjustment in adjustments:
                try:
                    agent = adjustment["agent"]
                    delta = float(adjustment["delta"])
                    reason = adjustment["reason"]
                    valid = (agent in allowed and agent not in seen and math.isfinite(delta)
                             and abs(delta) <= 0.25 and isinstance(reason, str) and bool(reason.strip()))
                except (KeyError, TypeError, ValueError):
                    valid = False
                if not valid:
                    raise AgentError("议长微调必须每角色最多一条、有证据且幅度不超过0.25")
                seen.add(agent)
            return str(data.get("summary", "")), adjustments
        evidence = "；".join(f"{m.name}：{m.text[:70]}" for m in messages)
        adjustments: list[dict[str, Any]] = []
        if user_input.meal_budget is None and user_input.weekly_spend > user_input.budget * 0.8:
            adjustments.append({"agent": "钱包", "delta": 0.1, "reason": "输入明确显示支出超过预算80%，在规则初权上做有限+0.1微调"})
        if not user_input.fridge.strip() or "空" in user_input.fridge:
            adjustments.append({"agent": "冰箱", "delta": 0.1, "reason": "输入明确显示冰箱为空，减少假设已有食材"})
        if user_input.vegetarian:
            adjustments.append({"agent": "营养师", "delta": 0.1, "reason": "素食作为硬约束，需要额外核对配料"})
        summary = f"议长总结（基于本轮发言证据）：{evidence or '暂无可用发言'}。微调仅按明确输入规则执行，幅度受限。"
        return summary, adjustments

    def extract_proposals(self, user_input: UserInput, messages: list[Message], summary: str) -> list[Proposal]:
        if not self._repairing_proposals:
            self.proposal_attempts = []
        if not self.use_mock:
            if self._proposal_payload is not None:
                data = {"proposals": self._proposal_payload}
                self._proposal_payload = None
            else:
                data = self._real_json(
                    "从输入和讨论中提取2到4个可执行候选。返回顶层 JSON 对象，严格使用"
                    ' {"proposals":[{"id":"P1","mode":"做饭或外卖","dish":"...","reason":"...",'
                    '"estimated_cost":0,"nutrition":"...","steps":["..."],"ingredients":["..."],"missing_ingredients":[]}]}。'
                    "ingredients必须列出全部实际食材和调味料；素食按纯素处理，不能含动物成分。"
                    "所有模式都禁止使用已过期食材；外卖步骤也不能加入过期库存。"
                    "储存未知或储存不明的剩饭不以气味判断可食，也不能作为任何候选的食材或步骤。"
                    "做饭只能把用户明确列出的食材视为已有；缺少的主食、配料或调味料必须列出采购步骤，采购金额计入estimated_cost，不能虚构库存。"
                    "候选要有实质不同的执行方式，不要只给同一做饭方案换写法。预算允许时至少提供一个可执行的外卖选项，费用包含配送估算且不超剩余预算；若无法做到，不得编造价格或食材。"
                    f"\n用户：{user_input.prompt_json()}\n总结：{summary}\n讨论：{json.dumps([m.model_dump(mode='json') for m in messages], ensure_ascii=False)}"
                )
            try:
                proposals = [Proposal(**item) for item in data["proposals"]]
            except Exception as exc:
                raise AgentError(f"议长候选JSON不合法：{exc}") from exc
            if not 2 <= len(proposals) <= 4 or len({p.id for p in proposals}) != len(proposals):
                raise AgentError("议长候选数量或ID不合法")
        else:
            proposals = proposal_seed(user_input)
        self.last_proposals = proposals
        # Hard constraints are applied before scores so a soft preference cannot override them.
        for proposal in proposals:
            if user_input.vegetarian and any(token in proposal.dish for token in ("鸡", "牛奶", "动物")):
                proposal.feasible = False
                proposal.constraint_note = "素食按保守纯素处理，排除含动物成分候选"
            full_text = " ".join([proposal.dish, proposal.reason, proposal.nutrition, *proposal.steps, *proposal.ingredients])
            animal_tokens = (
                "猪", "鱼", "牛", "鸡", "虾", "蟹", "羊", "鸭", "鹅", "兔", "鹌鹑", "肉",
                "蚝油", "鱼露", "虾酱", "牡蛎", "蛤蜊", "扇贝", "贻贝", "海参", "鲍", "鱿", "章鱼",
                "奶", "乳酪", "乳清", "酪蛋白", "黄油", "蛋黄", "蛋清", "蛋液", "蛋羹", "蛋饼",
                "明胶", "蜂蜜", "动物", "高汤", "骨汤",
            )
            # These explicit plant substitutes are not animal ingredients.
            vegan_text = re.sub(r"椰奶|豆奶|燕麦奶|杏仁奶|植物奶|素肉|植物肉", "植物替代品", full_text)
            bare_egg = any(ingredient.strip() == "蛋" for ingredient in proposal.ingredients)
            if user_input.vegetarian and (bare_egg or any(token in vegan_text for token in animal_tokens)):
                proposal.feasible = False
                proposal.constraint_note = "素食按保守纯素处理，候选文本或配料含动物成分"
            if proposal.mode == "做饭":
                ingredients_complete = bool(proposal.ingredients) and all(
                    _valid_ingredient_name(ingredient) for ingredient in proposal.ingredients
                )
                if not ingredients_complete:
                    proposal.feasible = False
                    proposal.constraint_note = "做饭方案必须列出完整、可核对的食材清单"
                inferred_missing: list[str] = []
                if user_input.shopping_willingness == "不愿意":
                    for ingredient in proposal.ingredients if ingredients_complete else []:
                        clean = re.sub(r"[（(].*?[）)]", "", ingredient).strip()
                        clean = re.sub(r"\d+(?:\.\d+)?\s*(?:个|克|g|毫升|ml|份)?", "", clean, flags=re.I).strip()
                        if clean and clean not in user_input.fridge:
                            inferred_missing.append(ingredient)
                proposal.missing_ingredients = list(dict.fromkeys([*proposal.missing_ingredients, *inferred_missing]))
                if user_input.cooking_willingness == "不愿意":
                    proposal.feasible = False
                    proposal.constraint_note = "用户不愿意做饭，做饭候选已过滤"
                elif proposal.missing_ingredients:
                    has_purchase_step = any(
                        token in step
                        for step in proposal.steps
                        for token in ("买", "采购", "购买", "补齐", "补充")
                    )
                    if not has_purchase_step or proposal.estimated_cost <= 0:
                        proposal.feasible = False
                        proposal.constraint_note = "菜谱缺料必须列出采购步骤并计入成本"
                    elif user_input.shopping_willingness == "不愿意":
                        proposal.feasible = False
                        proposal.constraint_note = "用户不愿意买菜，缺料做饭候选不可行"
            fridge_remainder = re.sub(r"\d+天(?:后)?过期", "", user_input.fridge)
            fridge_remainder = re.sub(r"尚未过期|未过期|没有过期|没过期", "", fridge_remainder)
            if proposal.mode == "做饭" and "过期" in fridge_remainder:
                # A conservative safety gate prevents an LLM or a template
                # from turning a fridge containing expired food into a meal.
                proposal.feasible = False
                proposal.constraint_note = "冰箱含明确已过期食材，做饭候选需先排除过期来源"
            # Apply to the candidate content regardless of its mode. Merely
            # labelling a meal takeout cannot authorize expired ingredients.
            content_expiry = _expiry_content(full_text)
            uses_unverified_stock = ("过期" in fridge_remainder and any(
                token in full_text for token in ("冰箱", "库存", "现有食材", "家里的", "家中的", "剩饭")
            ))
            if "过期" in content_expiry or uses_unverified_stock:
                proposal.feasible = False
                proposal.constraint_note = "候选内容涉及明确过期食材，所有模式均不可使用"
            unknown_storage = any(token in user_input.fridge for token in (
                "储存未知", "储存不明", "存储未知", "存储不明", "保存未知", "保存不明", "存放未知", "存放不明",
            ))
            leftover_used = (
                any(_explicitly_uses_leftover(item) for item in proposal.ingredients)
                or _explicitly_uses_leftover(proposal.dish)
                or any(_explicitly_uses_leftover(step) for step in proposal.steps)
            )
            if unknown_storage and leftover_used:
                proposal.feasible = False
                proposal.constraint_note = "剩饭储存未知，不以气味判断可食，候选不得使用"
            if proposal.estimated_cost > user_input.remaining_budget:
                proposal.feasible = False
                budget_label = "本餐预算" if user_input.meal_budget is not None else "剩余预算"
                proposal.constraint_note = f"估算成本 {proposal.estimated_cost:.0f} 超出{budget_label} {user_input.remaining_budget:.0f}"
            if user_input.remaining_budget == 0 and proposal.estimated_cost > 0:
                proposal.feasible = False
                proposal.constraint_note = "预算为0，付费候选不可行"
        if sum(p.feasible for p in proposals) < 2:
            self.proposal_attempts.append([p.model_dump(mode="json") for p in proposals])
            if not self.use_mock and not self._repairing_proposals:
                rejected = json.dumps([p.model_dump(mode="json") for p in proposals], ensure_ascii=False)
                self._repairing_proposals = True
                try:
                    repair_data = self._real_json(
                        "候选经过安全和意愿检查后少于两个可行方案。请只根据以下拒绝原因重新生成2到4个实质不同的候选，"
                        "不要复活被拒绝的做饭或缺料方案；可以生成预算内的外卖。仍严格返回"
                        '{"proposals":[{"id":"P1","mode":"做饭或外卖","dish":"...","reason":"...",'
                        '"estimated_cost":0,"nutrition":"...","steps":["..."],"ingredients":["..."],"missing_ingredients":[]}]}，'
                        "并满足用户的做饭/买菜意愿和本餐预算。"
                        f"\n用户：{user_input.prompt_json()}\n拒绝记录：" + rejected
                    )
                    repaired = repair_data.get("proposals")
                    if not isinstance(repaired, list):
                        raise AgentError("修复候选格式不合法")
                    self._proposal_payload = repaired
                    return self.extract_proposals(user_input, messages, summary)
                finally:
                    self._repairing_proposals = False
                    self._proposal_payload = None
            raise AgentError("硬约束后少于两个可执行候选；请补充预算、可用食材或修改输入后重试")
        return proposals

    def score_proposals(self, proposals: list[Proposal], votes: list[Vote]) -> dict[str, float]:
        """Utility aggregator; Parliament keeps individual votes as the source of truth."""
        totals: dict[str, float] = defaultdict(float)
        for vote in votes:
            for pid, score in vote.scores.items():
                totals[pid] += score
        return dict(totals)

    def decide(
        self,
        winner: Proposal,
        weighted_scores: dict[str, float],
        votes: list[Vote],
        disagreements: list[str],
    ) -> Recommendation:
        if not self.use_mock:
            data = self._real_json(
                "请将确定的获胜候选格式化为最终建议。返回顶层 JSON 对象，且严格包含以下七个字段："
                '{"recommendation":"做饭或外卖","dish":"...","reason":"...","estimated_cost":0,'
                '"nutrition":"...","steps":["..."],"disagreements":"..."}。'
                "不能改变候选的mode、dish、estimated_cost，也不能引入未给出的候选。"
                f"\n候选：{winner.model_dump_json(ensure_ascii=False)}\n分歧：{json.dumps(disagreements, ensure_ascii=False)}"
            )
            # The model call is used to format/check the decision, while every
            # factual field remains locked to the deterministic winner.
            return Recommendation(mode=winner.mode, dish=winner.dish, reason=winner.reason, estimated_cost=winner.estimated_cost, nutrition=winner.nutrition, steps=winner.steps, disagreements="；".join(disagreements))
        return Recommendation(
            mode=winner.mode,
            dish=winner.dish,
            reason=winner.reason + f"（加权主分 {weighted_scores.get(winner.id, 0):.2f}）",
            estimated_cost=round(float(winner.estimated_cost), 2),
            nutrition=winner.nutrition,
            steps=winner.steps,
            disagreements="；".join(disagreements),
        )
