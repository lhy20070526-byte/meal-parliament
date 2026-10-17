from .base import BaseAgent


class Lazy(BaseAgent):
    name, emoji = "懒惰", "🛋️"
    display_name, job_title = "周顾问", "时间与精力"
    citation_target_names = ("冰箱", "钱包", "营养师")
    system_prompt = "你是周顾问，负责温和地评估时间、步骤、清理成本和当下精力。请基于当前事实说2到3句中文，帮助用户选择更容易完成的一餐。"

    def mock_speak(self, user_input, context=""):
        late = user_input.hour >= 21 or user_input.hunger >= 8
        return f"现在{user_input.time}，饥饿度{user_input.hunger}/10；{ '优先十五分钟内完成、少清理。' if late else '三步以内更容易坚持。'}"

    def heuristic_score(self, proposal, user_input, context=""):
        if proposal.mode == "外卖":
            score = 8.5
        else:
            score = 6.0 if len(proposal.steps) <= 3 else 4.5
        if user_input.hunger >= 8 or user_input.hour >= 21:
            score += 1.0 if proposal.mode == "外卖" else -0.5
        return score

    def mock_debate(self, user_input, other, context=""):
        return f"引用 [{other.id}]：“{other.text[:18]}”。我饿{user_input.hunger}/10且走了{user_input.steps}步，洗切和清理不能凭空消失，优先15分钟内完成。"
