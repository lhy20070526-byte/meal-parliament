import unittest
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

from core.models import UserInput
from core.parliament import Parliament
from core.weights import calculate_weights
from agents.base import AgentError
from agents.nutritionist import Nutritionist
from core.models import Proposal


def scenario(name: str) -> UserInput:
    presets = {
        "late": dict(time="21:30", hunger=9, steps=8000, screen_hours=12, weekly_spend=800, budget=1000, fridge="鸡蛋（3天后过期）、剩饭、半颗白菜、临期牛奶", recent_diet="连续外卖，蔬菜少", taste="喜欢辣", special_needs="无"),
        "payday": dict(time="19:00", hunger=7, steps=5000, screen_hours=8, weekly_spend=200, budget=1500, fridge="空", recent_diet="正常", taste="日料", special_needs="无", wish="今天想吃顿好的"),
        "cut": dict(time="18:00", hunger=6, steps=15000, screen_hours=10, weekly_spend=300, budget=800, fridge="鸡胸肉、西兰花、糙米", recent_diet="高油高盐", taste="喜欢重口", special_needs="减脂"),
    }
    return UserInput(**presets[name])


class CoreTests(unittest.TestCase):
    def test_weight_rules_and_zero_budget_protection(self):
        user = UserInput(time="21:00", hunger=8, weekly_spend=1, budget=0, fridge="鸡蛋3天后过期、牛奶已过期", recent_diet="没吃蔬菜")
        weights = calculate_weights(user)
        self.assertEqual(weights["懒惰"]["value"], 2.0)
        self.assertEqual(weights["钱包"]["value"], 1.4)
        self.assertEqual(weights["冰箱"]["value"], 1.5)
        self.assertEqual(weights["营养师"]["value"], 1.5)

    def test_mock_trace_has_citations_and_complete_votes(self):
        result = Parliament(mode="mock").run(scenario("late"))
        debates = [m for m in result.messages if m.round == 2]
        self.assertEqual(len(debates), 5)
        self.assertTrue(all("[M" in m.text for m in debates))
        self.assertIn(len(result.votes), (5, 10))
        ids = {p.id for p in result.proposals}
        self.assertTrue(all(set(v.scores) == ids for v in result.votes))

    def test_presets_cover_extra_round_and_no_extra_round(self):
        results = [Parliament(mode="mock").run(scenario(name)) for name in ("late", "payday", "cut")]
        self.assertTrue(any(result.extra_round for result in results))
        self.assertTrue(any(not result.extra_round for result in results))

    def test_budget_is_hard_constraint_and_export_contract(self):
        result = Parliament(mode="mock").run(scenario("payday"))
        winner = next(p for p in result.proposals if p.id == result.winning_proposal_id)
        self.assertLessEqual(winner.estimated_cost, scenario("payday").remaining_budget)
        exported = result.export_dict()
        self.assertEqual(exported["recommendation"], result.recommendation.mode)
        self.assertIsInstance(exported["disagreements"], str)
        self.assertNotIn("mode", {k for k in exported if k in {"mode", "disagreements"}})
        self.assertIn("trace", exported)

    def test_gap_boundary_zero_and_vegetarian_hard_constraint(self):
        self.assertFalse(Parliament.should_extra_round(10.0, 9.0))
        self.assertTrue(Parliament.should_extra_round(0.0, 0.0))
        user = UserInput(time="18:00", hunger=5, budget=1200, weekly_spend=100, fridge="豆腐、西兰花、糙米", recent_diet="", taste="清淡", special_needs="素食")
        result = Parliament(mode="mock").run(user)
        winner = next(p for p in result.proposals if p.id == result.winning_proposal_id)
        self.assertNotRegex(winner.dish, r"鸡|牛奶|猪|牛肉")
        self.assertGreaterEqual(sum(p.feasible for p in result.proposals), 2)

    def test_willingness_is_in_fingerprint_and_unwilling_cooking_is_filtered(self):
        user = UserInput(meal_budget=40, fridge="", cooking_willingness="不愿意", shopping_willingness="不愿意")
        other = user.model_copy(update={"cooking_willingness": "愿意"})
        self.assertNotEqual(user.fingerprint, other.fingerprint)
        result = Parliament(mode="mock").run(user)
        self.assertTrue(all(p.mode == "外卖" for p in result.proposals if p.feasible))
        self.assertFalse(result.proposals[0].feasible)

    def test_zero_meal_budget_keeps_rejected_candidates_in_trace(self):
        events = []
        user = UserInput(meal_budget=0, fridge="", cooking_willingness="愿意", shopping_willingness="不愿意")
        with self.assertRaises(AgentError):
            Parliament(mode="mock").run(user, on_event=events.append)
        rejected = [event for event in events if event.get("kind") == "rejected_proposals"]
        self.assertEqual(len(rejected), 1)
        self.assertTrue(rejected[0]["proposals"])
        self.assertTrue(all("constraint_note" in item for item in rejected[0]["proposals"]))

    def test_meal_budget_replaces_legacy_weekly_budget_for_prompts(self):
        user = UserInput(meal_budget=35, budget=999, weekly_spend=998)
        self.assertEqual(user.remaining_budget, 35)
        prompt = user.prompt_json()
        self.assertIn('"meal_budget": 35.0', prompt)
        self.assertNotIn('"weekly_spend"', prompt)
        self.assertNotIn('"budget"', prompt)

    def test_real_adapter_rejects_missing_candidate_score(self):
        class FakeCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"scores":{"P1":5},"rationale":"缺少P2"}'))])

        class FakeClient:
            def __init__(self, api_key, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

        proposals = [
            Proposal(id="P1", mode="做饭", dish="A", reason="", estimated_cost=1, nutrition="", steps=["a"]),
            Proposal(id="P2", mode="外卖", dish="B", reason="", estimated_cost=1, nutrition="", steps=["b"]),
        ]
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-only", "PARLIAMENT_MODE": "real"}), patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeClient)}):
            with self.assertRaises(AgentError):
                Nutritionist(mode="real").vote(UserInput(), proposals)


if __name__ == "__main__":
    unittest.main()
