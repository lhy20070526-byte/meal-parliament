"""Synchronous, inspectable orchestration for the Meal Parliament demo."""

from __future__ import annotations

import math
import copy
from collections.abc import Callable
from typing import Any

from agents.fridge import Fridge
from agents.lazy import Lazy
from agents.nutritionist import Nutritionist
from agents.speaker import Speaker
from agents.taste import Taste
from agents.wallet import Wallet
from agents.base import AgentError, BaseAgent
from .models import Message, ParliamentResult, Proposal, UserInput, Vote
from .weights import calculate_weights


EventCallback = Callable[[dict[str, Any]], None]
RATIO_TOLERANCE = 1e-9


class Parliament:
    """Run all phases and retain the complete trace in memory and the result."""

    def __init__(self, *, mode: str | None = None) -> None:
        self.mode = mode
        self.trace: list[dict[str, Any]] = []
        self.agents: list[BaseAgent] = [Nutritionist(mode=mode), Wallet(mode=mode), Lazy(mode=mode), Fridge(mode=mode), Taste(mode=mode)]
        self.speaker = Speaker(mode=mode)

    def _event(self, kind: str, payload: dict[str, Any], callback: EventCallback | None) -> None:
        event = {"kind": kind, **payload}
        self.trace.append(event)
        if callback:
            callback(event)

    def _speak(self, agent: BaseAgent, context: str, others: list[Message], user: UserInput, round_number: int, callback: EventCallback | None) -> Message | None:
        last_error: str | None = None
        for attempt in range(1, 3):
            try:
                text = agent.speak(context, others, user)
                cited_id = cited_excerpt = None
                if round_number >= 2:
                    cited_id, cited_excerpt = self._validate_citation(text, agent, others)
                msg = Message(id=f"M{len([e for e in self.trace if e.get('kind') == 'message']) + 1}", role="specialist", emoji=agent.emoji, name=agent.name, text=text, weight=agent.weight, round=round_number)
                # Citation metadata is kept in trace so Message remains a stable public shape.
                self._event("message", {"message": msg.model_dump(mode="json"), "citation": {"id": cited_id, "excerpt": cited_excerpt} if cited_id else None}, callback)
                return msg
            except (AgentError, ValueError) as exc:
                last_error = str(exc)
                self._event("agent_retry", {"agent": agent.name, "round": round_number, "attempt": attempt, "error": last_error}, callback)
        error_message = Message(id=f"ERR-{agent.name}-{round_number}", role="specialist", emoji=agent.emoji, name=agent.name, text=f"错误：{last_error}（已保留当前trace，未静默切换mock）", weight=agent.weight, round=round_number, error=True)
        self._event("agent_error", {"message": error_message.model_dump(mode="json")}, callback)
        return None

    @staticmethod
    def _validate_citation(text: str, agent: BaseAgent, others: list[Message]) -> tuple[str, str]:
        """Require an ID and an exact >=8-character excerpt from another speaker."""
        import re

        match = re.search(r"\[([A-Za-z0-9_\-]+)\].*?[“\"](.{8,80}?)[”\"]", text, flags=re.S)
        if not match:
            raise ValueError("第二轮必须引用其他发言的消息ID和至少8字原文片段")
        message_id, excerpt = match.group(1), match.group(2)
        candidates = {m.id: m for m in others if m.name != agent.name}
        if message_id not in candidates:
            raise ValueError(f"引用消息 {message_id} 不存在或来自自己")
        if excerpt not in candidates[message_id].text:
            raise ValueError("引用片段与原发言不匹配")
        return message_id, excerpt

    @staticmethod
    def _apply_adjustments(weights: dict[str, dict[str, Any]], adjustments: list[dict[str, Any]]) -> None:
        # Validate the entire batch before mutating: one bounded adjustment
        # per role prevents repeated small deltas from bypassing the limit.
        if not isinstance(adjustments, list):
            raise ValueError("议长微调必须是列表")
        seen: set[str] = set()
        pending: list[tuple[str, float, float, str]] = []
        for item in adjustments:
            if not isinstance(item, dict):
                raise ValueError("议长微调格式错误")
            name = item.get("agent")
            if not isinstance(name, str) or name not in weights or name in seen:
                raise ValueError("议长微调角色必须有效且每角色最多一条")
            seen.add(name)
            try:
                delta = float(item["delta"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("议长微调缺少有效delta") from None
            reason = item.get("reason")
            if not math.isfinite(delta) or abs(delta) > 0.25 or not isinstance(reason, str) or not reason.strip():
                raise ValueError("议长微调必须有证据且绝对幅度不超过0.25")
            old = float(weights[name]["value"])
            new = round(old + delta, 4)
            if not math.isfinite(new) or new <= 0:
                raise ValueError(f"非法权重调整：{name}={new}")
            pending.append((name, new, delta, reason))
        for name, new, delta, reason in pending:
            weights[name]["value"] = new
            weights[name]["reasons"].append(reason)
            weights[name]["adjustment"] = delta

    @staticmethod
    def _aggregate(votes: list[Vote], weights: dict[str, dict[str, Any]], proposals: list[Proposal]) -> dict[str, float]:
        ids = [p.id for p in proposals]
        expected = {"营养师", "钱包", "懒惰", "冰箱", "口味"}
        if len(votes) != len(expected) or {vote.agent for vote in votes} != expected or set(weights) != expected:
            raise ValueError("投票必须包含五个不同专家各一张票及对应权重")
        if len({vote.round for vote in votes}) != 1 or votes[0].round not in {1, 2}:
            raise ValueError("投票必须来自同一有效轮次")
        if not 2 <= len(ids) <= 4 or len(ids) != len(set(ids)):
            raise ValueError("候选必须为2到4个且ID唯一")
        totals = {pid: 0.0 for pid in ids}
        for vote in votes:
            if set(vote.scores) != set(ids):
                raise ValueError(f"{vote.agent}评分候选集合不完整")
            weight = float(weights[vote.agent]["value"])
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"{vote.agent}权重必须为有限正数")
            for pid in ids:
                score = float(vote.scores[pid])
                if not math.isfinite(score) or not (0 <= score <= 10):
                    raise ValueError(f"{vote.agent}对{pid}的分数非法")
                totals[pid] += weight * score
        return {pid: round(value, 6) for pid, value in totals.items()}

    @staticmethod
    def _rank(scores: dict[str, float], proposals: list[Proposal]) -> list[str]:
        feasible = {p.id for p in proposals if p.feasible}
        return sorted(scores, key=lambda pid: (pid not in feasible, -scores[pid], pid))

    @staticmethod
    def _disagreements(votes: list[Vote], proposals: list[Proposal]) -> list[str]:
        focus = {
            "营养师": "营养均衡、蛋白质和油盐",
            "钱包": "预算与新增成本",
            "懒惰": "耗时、步骤和清理",
            "冰箱": "库存真实性与采购",
            "口味": "口味满足感",
        }
        out: list[str] = []
        for proposal in proposals:
            vals = [vote.scores[proposal.id] for vote in votes]
            if vals and max(vals) - min(vals) >= 3:
                # max/min preserve the supplied expert order on a tie.  That
                # makes the explanation stable without inventing a new tie
                # breaker in the scoring path.
                high = max(votes, key=lambda vote: vote.scores[proposal.id])
                low = min(votes, key=lambda vote: vote.scores[proposal.id])
                high_focus = focus.get(high.agent, "角色约束")
                low_focus = focus.get(low.agent, "角色约束")
                out.append(
                    f"{proposal.id}（{proposal.dish}）："
                    f"{high.emoji}{high.agent}给{high.scores[proposal.id]:.1f}分，偏重{high_focus}；"
                    f"{low.emoji}{low.agent}给{low.scores[proposal.id]:.1f}分，偏重{low_focus}；"
                    f"分歧在于{high_focus}与{low_focus}的取舍。"
                )
        return out or ["各角色分歧未达到3分，仍保留完整投票trace供复核"]

    def _normalized(self, scores: dict[str, float], proposals: list[Proposal]) -> dict[str, float]:
        feasible_ids = [p.id for p in proposals if p.feasible]
        if not feasible_ids:
            return {pid: 0.0 for pid in scores}
        # Each candidate receives one score from every role, so divide by the
        # total role weight to expose a 0-10 comparable score. This is an
        # auxiliary view; ranking still uses the weighted sum.
        role_weight = sum(float(item["value"]) for item in self._last_weights.values()) if hasattr(self, "_last_weights") else 0.0
        if role_weight <= 0:
            return {pid: 0.0 for pid in scores}
        return {pid: round(scores[pid] / role_weight, 2) if pid in feasible_ids else 0.0 for pid in scores}

    @staticmethod
    def should_extra_round(top_score: float, second_score: float | None) -> bool:
        """Use a strict 10% boundary; an exact 10% lead does not trigger."""
        if second_score is None or not math.isfinite(top_score) or not math.isfinite(second_score):
            return False
        if top_score <= 0:
            return True
        ratio = (top_score - second_score) / max(top_score, 1e-12)
        return ratio + RATIO_TOLERANCE < 0.10

    def run(self, user_input: UserInput, on_event: EventCallback | None = None) -> ParliamentResult:
        self.trace = []
        weights = calculate_weights(user_input)
        self._last_weights = weights
        for agent in self.agents:
            agent.weight = weights[agent.name]["value"]
        self._event("initialized", {"input_fingerprint": user_input.fingerprint, "mode": "mock" if self.agents[0].use_mock else "real", "weights": copy.deepcopy(weights)}, on_event)

        messages: list[Message] = []
        for agent in self.agents:
            msg = self._speak(agent, "首轮发言：只依据用户输入提出角色判断。", [], user_input, 1, on_event)
            if msg is None:
                raise AgentError(f"{agent.name}首轮发言失败，已停止以避免伪造完整流程")
            messages.append(msg)

        summary, adjustments = self.speaker.summarize(user_input, messages, weights)
        self._apply_adjustments(weights, adjustments)
        for agent in self.agents:
            agent.weight = weights[agent.name]["value"]
        self._event("summary", {"text": summary, "adjustments": adjustments}, on_event)

        debate_messages: list[Message] = []
        for agent in self.agents:
            others = [m for m in messages if m.name != agent.name]
            context = "第二轮辩论：必须在回答中引用一个其他专家的消息ID，例如 [M1]，并逐字引用至少8个字符。"
            msg = self._speak(agent, context, others, user_input, 2, on_event)
            if msg is None:
                raise AgentError(f"{agent.name}第二轮引用校验失败，已停止以避免伪造完整流程")
            debate_messages.append(msg)
        all_messages = messages + debate_messages

        try:
            proposals = self.speaker.extract_proposals(user_input, all_messages, summary)
        except AgentError as exc:
            rejected = self.speaker.proposal_attempts or [[proposal.model_dump(mode="json") for proposal in self.speaker.last_proposals]]
            self._event(
                "rejected_proposals",
                {"attempts": rejected, "proposals": rejected[-1] if rejected else [], "error": str(exc)},
                on_event,
            )
            raise
        if self.speaker.proposal_attempts:
            self._event(
                "rejected_proposals",
                {
                    "attempts": self.speaker.proposal_attempts,
                    "proposals": self.speaker.proposal_attempts[-1],
                    "error": "首次生成少于两个可行方案，已重新整理",
                },
                on_event,
            )
        self._event("proposals", {"proposals": [p.model_dump(mode="json") for p in proposals]}, on_event)
        votes: list[Vote] = []
        discussion = "\n".join(f"{m.id} {m.name}: {m.text}" for m in all_messages)
        for agent in self.agents:
            vote = agent.vote(user_input, proposals, context=f"讨论证据（不含任何分数或排名）：\n{discussion}\n只看用户输入和候选全文；禁止读取其他专家分数或排名。", round_number=1)
            votes.append(vote)
            self._event("vote", {"vote": vote.model_dump(mode="json")}, on_event)
        weighted_scores = self._aggregate(votes, weights, proposals)
        order = self._rank(weighted_scores, proposals)
        feasible_order = [pid for pid in order if next(p for p in proposals if p.id == pid).feasible]
        if not feasible_order:
            raise AgentError("所有候选都违反预算或硬约束，无法生成安全推荐")
        top = feasible_order[0]
        second = feasible_order[1] if len(feasible_order) > 1 else None
        ratio = 0.0 if second is None else (weighted_scores[top] - weighted_scores[second]) / max(weighted_scores[top], 1e-12)
        extra_round = self.should_extra_round(weighted_scores[top], weighted_scores[second] if second else None)
        self._event("score", {"round": 1, "weighted_scores": weighted_scores, "normalized_scores": self._normalized(weighted_scores, proposals), "gap_ratio": ratio, "extra_round": extra_round}, on_event)

        final_votes = votes
        if extra_round:
            # The candidate set and weights are frozen; only an independent re-vote is added.
            extra_discussion: list[Message] = []
            for agent in self.agents:
                others = [m for m in all_messages + extra_discussion if m.name != agent.name]
                msg = self._speak(agent, "第三轮加赛辩论：必须引用其他专家消息ID和至少8字原文，再给出新的独立判断。", others, user_input, 3, on_event)
                if msg is None:
                    raise AgentError(f"{agent.name}加赛辩论失败，已停止以避免伪造完整流程")
                extra_discussion.append(msg)
            all_messages.extend(extra_discussion)
            frozen_discussion = "\n".join(f"{m.id} {m.name}: {m.text}" for m in all_messages)
            extra_votes: list[Vote] = []
            for agent in self.agents:
                vote = agent.vote(user_input, proposals, context=f"加赛讨论证据（不含任何分数或排名）：\n{frozen_discussion}\n重新独立评估全部候选，不传入其他专家分数、排名或汇总。", round_number=2)
                extra_votes.append(vote)
                votes.append(vote)
                self._event("vote", {"vote": vote.model_dump(mode="json")}, on_event)
            final_votes = extra_votes
            weighted_scores = self._aggregate(final_votes, weights, proposals)
            order = self._rank(weighted_scores, proposals)
            feasible_order = [pid for pid in order if next(p for p in proposals if p.id == pid).feasible]
            top = feasible_order[0]
            self._event("score", {"round": 2, "weighted_scores": weighted_scores, "normalized_scores": self._normalized(weighted_scores, proposals), "candidate_weights_frozen": True}, on_event)

        winner = next(p for p in proposals if p.id == top)
        recommendation = self.speaker.decide(winner, weighted_scores, final_votes, self._disagreements(final_votes, proposals))
        self._event("final", {"winning_proposal_id": winner.id, "recommendation": recommendation.model_dump(mode="json")}, on_event)
        return ParliamentResult(
            input_fingerprint=user_input.fingerprint,
            weights=weights,
            messages=all_messages,
            summary=summary,
            adjustments=adjustments,
            proposals=proposals,
            votes=votes,
            weighted_scores=weighted_scores,
            normalized_scores=self._normalized(weighted_scores, proposals),
            winning_proposal_id=winner.id,
            extra_round=extra_round,
            recommendation=recommendation,
            trace=self.trace,
        )
