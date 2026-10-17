from .base import BaseAgent


class Wallet(BaseAgent):
    name, emoji = "钱包", "💰"
    display_name, job_title = "钱顾问", "餐费预算"
    citation_target_names = ("营养师", "口味", "懒惰")
    system_prompt = "你是钱顾问，负责温和而清楚地说明这一餐的预算、配送费和成本口径。只基于当前事实说2到3句中文，不能用偏好覆盖预算限制。"

    def mock_speak(self, user_input, context=""):
        label = "这顿最多可花" if user_input.meal_budget is not None else "按现有记录还可用"
        return f"{label}{user_input.remaining_budget:.0f}元；我会把配送费和补买食材一起算清楚。"

    def heuristic_score(self, proposal, user_input, context=""):
        if user_input.remaining_budget <= 0:
            return 0.0 if proposal.estimated_cost > 0 else 10.0
        ratio = proposal.estimated_cost / user_input.remaining_budget
        if ratio > 1:
            return 0.0
        if ratio <= 0.2:
            return 9.0
        if ratio <= 0.5:
            return 8.0
        if ratio <= 0.8:
            return 6.5
        return 4.0

    def mock_debate(self, user_input, other, context=""):
        return f"引用 [{other.id}]：“{other.text[:18]}”。补蛋白没问题，但这顿最多{user_input.remaining_budget:.0f}元，我会优先预算内的方案。"
