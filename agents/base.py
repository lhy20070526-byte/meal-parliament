"""Shared OpenAI adapter and deterministic specialist interface."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

from core.llm_config import LLMConfigError, create_client, get_llm_config
from core.models import Message, Proposal, UserInput, Vote


class AgentError(RuntimeError):
    """A visible, actionable model or validation error."""


class BaseAgent:
    name = "专家"
    display_name = "专家"
    job_title = "观察员"
    emoji = "🔎"
    citation_target_names: tuple[str, ...] = ()
    system_prompt = "请基于用户输入给出可核验的用餐建议。"
    shared_safety_boundary = "安全边界：储存未知的剩饭不以气味判断可食，也不作为候选食材。"
    shared_locale_guidance = "请使用简体中文，按中国大陆常见食材和家常做法表达；金额使用人民币元，价格只做估计，不编造实时外卖价格。"

    def __init__(self, weight: float = 1.0, *, mode: str | None = None) -> None:
        self.weight = weight
        self.history: list[dict[str, Any]] = []
        self.llm_config = get_llm_config()
        self.mode = (mode or self.llm_config.parliament_mode).lower()
        self.model = self.llm_config.model
        self._client: Any | None = None

    @property
    def use_mock(self) -> bool:
        # Explicit real mode must surface a useful missing-key error.  Auto
        # mode remains convenient for the offline demo and mocks only when the
        # selected provider is unconfigured.
        return self.mode in {"mock", "demo"} or (
            self.mode == "auto"
            and self.llm_config.provider in {"openai", "deepseek"}
            and not self.llm_config.configured
        )

    def _real_client(self) -> Any:
        if self.llm_config.provider not in {"openai", "deepseek"}:
            raise AgentError("LLM_PROVIDER 必须是 openai 或 deepseek")
        if not self.llm_config.configured:
            raise AgentError(f"真实模式未配置 {self.llm_config.provider} API key，请编辑 ~/.config/meal-parliament/.env")
        if self._client is None:
            try:
                self._client = create_client(self.llm_config)
            except LLMConfigError as exc:
                raise AgentError(str(exc)) from exc
        return self._client

    def _request_options(self) -> dict[str, Any]:
        return {"timeout": 60, **self.llm_config.request_options}

    def build_prompt(self, user_input: UserInput, context: str = "") -> str:
        return (
            f"用户输入：{user_input.prompt_json()}\n"
            f"任务上下文：{context}\n"
            f"{self.shared_locale_guidance}\n"
            f"{self.shared_safety_boundary}\n"
            "请只陈述与你的角色相关的事实、取舍和建议。"
        )

    def mock_speak(self, user_input: UserInput, context: str = "") -> str:
        return f"{self.name}：根据当前输入，建议把可核验约束写进选择理由。"

    def mock_debate(self, user_input: UserInput, other: Message, context: str = "") -> str:
        excerpt = other.text[:18]
        return f"我回应{other.name}：引用 [{other.id}]：“{excerpt}”。这条证据与我的角色约束有关，我保留自己的判断。"

    def speak(
        self,
        context: str,
        other_messages: Iterable[Message],
        user_input: UserInput,
    ) -> str:
        """Speak independently; a real call never silently falls back to mock."""
        if self.use_mock:
            if ("第二轮" in context or "引用" in context) and other_messages:
                available = list(other_messages)
                other = next(
                    (item for target in self.citation_target_names for item in available if item.name == target),
                    available[0],
                )
                text = self.mock_debate(user_input, other, context)
            else:
                text = self.mock_speak(user_input, context)
            self.history.append({"kind": "speak", "mode": "mock", "text": text})
            return text
        try:
            cited = "\n".join(f"[{m.id}] {m.name}: {m.text}" for m in other_messages)
            prompt = self.build_prompt(user_input, f"{context}\n可引用的其他发言：\n{cited}")
            client = self._real_client()
            options = self._request_options()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.3,
                **options,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise AgentError("模型返回了空发言")
            self.history.append({"kind": "speak", "mode": "real", "text": text})
            return text
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(f"{self.name}真实模式调用失败：{type(exc).__name__}，请检查密钥、网络或重试") from exc

    def heuristic_score(self, proposal: Proposal, user_input: UserInput, context: str = "") -> float:
        """Subclasses provide an independent score without seeing other votes."""
        return 5.0

    def vote(self, user_input: UserInput, proposals: list[Proposal], context: str = "", round_number: int = 1) -> Vote:
        if not proposals:
            raise AgentError("没有可评分候选")
        if not self.use_mock:
            try:
                ids = [p.id for p in proposals]
                prompt = (
                    f"用户输入：{user_input.prompt_json()}\n候选："
                    f"{json.dumps([p.model_dump(mode='json') for p in proposals], ensure_ascii=False)}\n"
                    f"讨论证据：{context}\n"
                    f"{self.shared_locale_guidance}\n"
                    f"{self.shared_safety_boundary}\n"
                    "请只按你的角色独立评分，不读取或猜测其他专家分数。只返回JSON："
                    '{"scores":{"P1":0},"rationale":"..."}，scores必须包含且仅包含每个候选ID，分数为0到10。'
                )
                options = self._request_options()
                response = self._real_client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    **options,
                )
                raw = (response.choices[0].message.content or "").strip()
                data = json.loads(raw)
                scores = data.get("scores")
                if set(scores or {}) != set(ids):
                    raise AgentError(f"真实评分候选集合不完整，应为 {ids}")
                checked = {pid: float(scores[pid]) for pid in ids}
                if any(not math.isfinite(value) or not 0 <= value <= 10 for value in checked.values()):
                    raise AgentError("真实评分包含非有限数或超出0到10")
                return Vote(agent=self.name, emoji=self.emoji, round=round_number, scores=checked, rationale=str(data.get("rationale", "")))
            except AgentError:
                raise
            except Exception as exc:
                raise AgentError(f"{self.name}真实评分失败：{type(exc).__name__}，请检查JSON、密钥或重试") from exc
        scores = {p.id: max(0.0, min(10.0, round(self.heuristic_score(p, user_input, context), 2))) for p in proposals}
        rationale = self.vote_rationale(user_input, proposals, scores, round_number)
        return Vote(agent=self.name, emoji=self.emoji, round=round_number, scores=scores, rationale=rationale)

    def vote_rationale(self, user_input: UserInput, proposals: list[Proposal], scores: dict[str, float], round_number: int) -> str:
        """Explain the local scoring choice in the voice of the role.

        The mock vote is deliberately still an independent vote, so this text
        must describe the role's criterion rather than pretend that a global
        winner was established.  Keep tie handling deterministic while making
        a tie visible to the reader.
        """
        if not scores:
            return f"第{round_number}轮没有可评分候选；我只按自己的角色约束发言。"
        ordered_ids = [proposal.id for proposal in proposals if proposal.id in scores]
        if not ordered_ids:
            ordered_ids = list(scores)
        best_score = max(scores[proposal_id] for proposal_id in ordered_ids)
        best_ids = [proposal_id for proposal_id in ordered_ids if math.isclose(scores[proposal_id], best_score, abs_tol=1e-9)]
        best = best_ids[0]
        candidate = next((proposal for proposal in proposals if proposal.id == best), None)
        dish = candidate.dish if candidate else best

        focus = {
            "营养师": "营养均衡、蛋白质和油盐",
            "钱包": "剩余预算与新增成本",
            "懒惰": "完成时间、步骤和清理负担",
            "冰箱": "库存是否真实可用、是否需要采购",
            "口味": "明确口味和吃饭满足感",
        }.get(self.name, "本角色的约束")
        if self.name == "钱包":
            budget_label = "本餐预算" if user_input.meal_budget is not None else "剩余预算"
            detail = f"{budget_label}{user_input.remaining_budget:.0f}元，成本{candidate.estimated_cost:.0f}元" if candidate else f"{budget_label}{user_input.remaining_budget:.0f}元"
        elif self.name == "懒惰":
            detail = f"{len(candidate.steps)}步、{candidate.mode}" if candidate else "执行步骤"
        elif self.name == "冰箱":
            detail = "候选标为可执行" if candidate and candidate.feasible else "候选存在库存或安全限制"
        elif self.name == "营养师":
            detail = candidate.nutrition[:32] if candidate else "营养标注"
        elif self.name == "口味":
            detail = user_input.taste or user_input.wish or "没有额外口味偏好"
        else:
            detail = "候选约束"
        tie_note = "并列最高，按候选ID顺序记录" if len(best_ids) > 1 else ""
        return (
            f"第{round_number}轮我按{focus}独立评分：{best}「{dish}」{best_score:.1f}分，"
            f"依据是{detail}。{tie_note}"
        ).strip()
