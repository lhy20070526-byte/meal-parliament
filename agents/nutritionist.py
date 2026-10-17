from .base import BaseAgent


class Nutritionist(BaseAgent):
    name, emoji = "营养师", "🥗"
    display_name, job_title = "林营养师", "营养搭配"
    citation_target_names = ("冰箱", "口味", "懒惰")
    system_prompt = "你是林营养师，负责用温和清楚的话说明营养搭配。只评估营养平衡、蔬菜、蛋白质、油盐和特殊需求；基于当前饮食事实说2到3句中文，不做医学诊断。"

    def mock_speak(self, user_input, context=""):
        need = user_input.special_needs if user_input.special_needs != "无" else "营养均衡"
        recent = user_input.recent_diet or ""
        if any(token in recent for token in ("蔬菜少", "蔬菜不足", "少蔬菜", "连续外卖", "没吃蔬菜")):
            return f"连续外卖、蔬菜又少，我先把一份青菜和蛋白质抢回来；今晚按{need}来。"
        if any(token in recent for token in ("高油", "高盐", "重口")):
            return f"这几天油盐已经偏高，今晚收一收；按{need}补一份清楚的蛋白质和蔬菜。"
        return f"最近饮食看起来还算正常，今晚保持一份蛋白质、一份蔬菜和合适主食；特殊需求按{need}执行。"

    def heuristic_score(self, proposal, user_input, context=""):
        text = proposal.dish + proposal.nutrition + proposal.reason
        score = 5.5
        if any(x in text for x in ("西兰花", "蔬菜", "蛋白", "鸡胸", "豆腐")):
            score += 2.0
        if "减脂" in user_input.special_needs and any(x in text for x in ("低油", "蔬菜", "鸡胸")):
            score += 1.5
        if "增肌" in user_input.special_needs and any(x in text for x in ("蛋白", "鸡胸", "豆腐")):
            score += 1.0
        if "素食" in user_input.special_needs and any(x in text for x in ("鸡胸", "鸡蛋", "牛奶")):
            score -= 8
        return score

    def mock_debate(self, user_input, other, context=""):
        return f"引用 [{other.id}]：“{other.text[:18]}”。我支持补足蔬菜和蛋白质，但高油高盐和过期食材先出局。"
