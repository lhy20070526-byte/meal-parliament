from .base import BaseAgent


class Fridge(BaseAgent):
    name, emoji = "冰箱", "🧊"
    display_name, job_title = "陈管家", "食材管理"
    citation_target_names = ("懒惰", "营养师", "钱包")
    system_prompt = "你是陈管家，负责谨慎而清楚地管理食材。只使用用户明确列出的食材；区分已过期、临期和储存未知，用当前盘点事实说2到3句中文，不臆造食材。"

    def mock_speak(self, user_input, context=""):
        fridge = user_input.fridge.strip() or "未填写/视为空"
        return f"我只认冰箱里真正写出来的东西：{fridge}。过期的扔掉，储存不明的剩饭先别逞强。"

    def heuristic_score(self, proposal, user_input, context=""):
        text = proposal.dish + proposal.reason
        if proposal.mode == "做饭" and proposal.feasible:
            score = 8.5
            if "鸡胸" in user_input.fridge and "鸡胸" in text:
                score += 1
            if "鸡蛋" in user_input.fridge and "鸡蛋" in text:
                score += 1
        elif not user_input.fridge.strip() and proposal.mode == "外卖":
            score = 8.5
        else:
            score = 4.5
        if "过期" in user_input.fridge and proposal.mode == "做饭":
            score -= 2
        return score

    def mock_debate(self, user_input, other, context=""):
        fridge = user_input.fridge.strip()
        if not fridge or "空" in fridge or "没有" in fridge:
            point = "冰箱是空的，别把鸡蛋白菜写进今晚菜单；外卖配料要现场核对。"
        elif all(token in fridge for token in ("鸡胸", "西兰花", "糙米")):
            point = "鸡胸、西兰花、糙米都在，做一盘是库存事实；别额外采购一堆新食材。"
        elif all(token in fridge for token in ("鸡蛋", "白菜", "剩饭")):
            point = "鸡蛋白菜和剩饭都列出来了，但剩饭储存未知，先确认安全再动锅。"
        else:
            point = f"我只确认这份库存：{fridge}；缺项不能当成已有，做饭候选要列采购。"
        return f"引用 [{other.id}]：“{other.text[:18]}”。{point}"
