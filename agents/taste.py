from .base import BaseAgent


class Taste(BaseAgent):
    name, emoji = "口味", "🌶️"
    display_name, job_title = "唐顾问", "饮食偏好"
    citation_target_names = ("营养师", "钱包", "懒惰")
    system_prompt = "你是唐顾问，负责记住用户明确说出的口味和今晚想吃什么。只评估饮食偏好与满足感，用当前事实说2到3句中文，尊重用户的选择。"

    def mock_speak(self, user_input, context=""):
        return f"我记得你要的是“{user_input.taste or '有滋有味'}”，清淡到没欲望可不算完成；在预算里给它一点脾气。"

    def heuristic_score(self, proposal, user_input, context=""):
        text = proposal.dish
        score = 6.0
        if "辣" in user_input.taste and "麻辣" in text:
            score += 3
        if "重口" in user_input.taste and any(x in text for x in ("麻辣", "辣")):
            score += 2
        if "日料" in user_input.wish + user_input.taste and "日式" in text:
            score += 3
        if "今天想吃顿好的" in user_input.wish and proposal.mode == "外卖":
            score += 1
        if "减脂" in user_input.special_needs and "麻辣" in text:
            score -= 2
        return score

    def mock_debate(self, user_input, other, context=""):
        return f"引用 [{other.id}]：“{other.text[:18]}”。我不接受清淡到没欲望；在预算和特殊需求允许时，用少量{user_input.taste or '调味'}保留满足感。"
