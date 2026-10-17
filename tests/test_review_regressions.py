"""Regression cases from the Astra core review; no external API calls."""
import copy
import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from core.models import Proposal, UserInput, Vote
from core.parliament import Parliament
from core.weights import calculate_weights
from agents.base import AgentError
from agents.speaker import Speaker
from agents.fridge import Fridge
from core.mock_data import proposal_seed


def proposal(pid, dish='豆腐蔬菜碗', *, mode='做饭', cost=5, steps=None, ingredients=None):
    return dict(id=pid, mode=mode, dish=dish, estimated_cost=cost,
                reason='常规用餐方案', nutrition='植物蛋白和蔬菜',
                steps=steps or ['制作一份'], ingredients=ingredients or ['豆腐', '蔬菜'])


class ReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'OPENAI_API_KEY': 'local-test-placeholder'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.speaker = Speaker(mode='real')

    def extract(self, user, proposals):
        with patch.object(self.speaker, '_real_json', return_value={'proposals': proposals}):
            return self.speaker.extract_proposals(user, [], '')

    def test_vegan_rejects_duck_goose_and_oyster_sauce_in_all_fields(self):
        candidates = [proposal('P1', '烤鸭饭'),
                      proposal('P2', steps=['加入蚝油和鹅肉']),
                      proposal('P3'), proposal('P4', ingredients=['豆腐', '椰奶'])]
        result = self.extract(UserInput(special_needs='素食'), candidates)
        self.assertEqual([p.feasible for p in result], [False, False, True, True])

    def test_takeout_cannot_add_expired_fridge_food(self):
        for instruction in ('将冰箱里已过期牛奶倒入套餐', '加入过期麦片', '搭配冰箱里的牛奶'):
            with self.subTest(instruction=instruction):
                candidates = [proposal('P1', mode='外卖', steps=[instruction]),
                              proposal('P2', mode='外卖'), proposal('P3', mode='外卖')]
                result = self.extract(UserInput(fridge='牛奶已过期、鸡蛋3天后过期'), candidates)
                self.assertEqual([p.feasible for p in result], [False, True, True])

    def test_unknown_stored_leftovers_are_not_a_candidate_ingredient(self):
        candidates = [
            proposal('P1', dish='鸡蛋白菜剩饭拌饭', ingredients=['鸡蛋', '白菜', '剩饭']),
            proposal('P2', mode='外卖'),
            proposal('P3', mode='外卖'),
        ]
        result = self.extract(UserInput(fridge='鸡蛋、白菜、剩饭（储存未知）'), candidates)
        self.assertFalse(result[0].feasible)
        self.assertIn('储存未知', result[0].constraint_note)

    def test_unknown_stored_leftover_safety_note_is_not_treated_as_use(self):
        safe = proposal('P1', dish='现做蔬菜套餐', steps=['丢弃剩饭', '选择现做套餐'])
        safe['reason'] = '避免使用储存未知剩饭，改用现做套餐'
        candidates = [
            safe,
            proposal('P2', mode='外卖'),
            proposal('P3', mode='外卖'),
        ]
        result = self.extract(UserInput(fridge='剩饭（储存未知）'), candidates)
        self.assertTrue(all(p.feasible for p in result))

    def test_unknown_stored_leftover_explicit_step_use_is_rejected(self):
        candidates = [
            proposal('P1', dish='现做蔬菜套餐', steps=['加入剩饭拌匀']),
            proposal('P2', mode='外卖'),
            proposal('P3', mode='外卖'),
        ]
        result = self.extract(UserInput(fridge='剩饭（储存未知）'), candidates)
        self.assertFalse(result[0].feasible)

    def test_live_candidate_without_leftover_in_name_remains_feasible(self):
        # The real DeepSeek diagnostic used this dish name; it contains an
        # explicit exclusion, not an instruction to consume the stored rice.
        safe = proposal('P1', dish='辣味白菜鸡蛋汤面（无剩饭版）',
                        steps=['采购面条', '煮面条并加入白菜鸡蛋'],
                        ingredients=['面条', '白菜', '鸡蛋'])
        safe['reason'] = '不碰储存未知的剩饭，缺少的面条先采购'
        result = self.extract(UserInput(fridge='鸡蛋、白菜、剩饭（储存未知）'),
                              [safe, proposal('P2', mode='外卖')])
        self.assertTrue(all(p.feasible for p in result))

    def test_leftover_version_label_is_not_an_exclusion(self):
        candidates = [
            proposal("P1", dish="鸡蛋炒饭（剩饭版）", ingredients=["鸡蛋", "剩饭版炒饭"], steps=["将上述食材炒熟"]),
            proposal("P2", mode="外卖"),
            proposal("P3", mode="外卖"),
        ]
        result = self.extract(UserInput(fridge="鸡蛋、剩饭（储存未知）"), candidates)
        self.assertFalse(result[0].feasible)

    def test_cooking_candidate_must_list_ingredients_when_shopping_is_unwilling(self):
        for invalid in ([], [""], ["  "], ["3个"], ["（）"]):
            with self.subTest(invalid=invalid):
                missing_ingredients = proposal("P1", dish="鸡蛋炒饭", steps=["加入鸡蛋和米饭炒熟"])
                missing_ingredients["ingredients"] = invalid
                candidates = [missing_ingredients, proposal("P2", mode="外卖"), proposal("P3", mode="外卖")]
                result = self.extract(UserInput(fridge="空", shopping_willingness="不愿意"), candidates)
                self.assertFalse(result[0].feasible)
                self.assertIn("食材清单", result[0].constraint_note)

        parenthetical_only = proposal("P1", dish="鸡蛋炒饭", steps=["加入鸡蛋和米饭炒熟"])
        parenthetical_only["ingredients"] = ["（少许）"]
        result = self.extract(UserInput(fridge="空", shopping_willingness="不愿意"), [
            parenthetical_only, proposal("P2", mode="外卖"), proposal("P3", mode="外卖")
        ])
        self.assertFalse(result[0].feasible)

    def test_unrelated_negation_cannot_hide_leftover_use(self):
        cases = [
            {'steps': ['不使用牛奶', '加入剩饭拌匀']},
            {'steps': ['不使用牛奶但加入剩饭拌匀']},
            {'ingredients': ['不使用牛奶', '剩饭']},
            {'steps': ['丢弃剩饭', '随后加入剩饭拌匀']},
            {'steps': ['将剩饭加热后拌入白菜']},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                result = self.extract(UserInput(fridge='剩饭（储存未知）'),
                                      [proposal('P1', **fields), proposal('P2', mode='外卖'), proposal('P3', mode='外卖')])
                self.assertFalse(result[0].feasible)

    def test_expiry_safety_note_does_not_cancel_real_expired_ingredient(self):
        safe = proposal('P1', mode='外卖', steps=['不使用过期食材', '选择新做套餐'])
        safe['reason'] = '禁止使用已过期食材'
        user = UserInput(fridge='鸡蛋3天后过期、剩饭（储存未知）')
        result = self.extract(user, [safe, proposal('P2', mode='外卖')])
        self.assertTrue(all(p.feasible for p in result))
        for steps in (['不使用过期食材', '加入过期麦片'], ['不使用牛奶但加入过期麦片']):
            result = self.extract(user, [proposal('P1', mode='外卖', steps=steps),
                                         proposal('P2', mode='外卖'), proposal('P3', mode='外卖')])
            self.assertFalse(result[0].feasible)

    def test_real_json_prompts_keep_json_contract_for_extract_and_decide(self):
        calls = []
        proposals = [proposal('P1'), proposal('P2', mode='外卖')]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    content = '{"proposals":' + json.dumps(proposals, ensure_ascii=False) + '}'
                else:
                    content = '{"recommendation":"做饭","dish":"豆腐蔬菜碗","reason":"证据","estimated_cost":5,"nutrition":"均衡","steps":["制作"],"disagreements":""}'
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        user = UserInput(fridge='豆腐、蔬菜')
        with patch.object(self.speaker, '_real_client', return_value=fake_client):
            self.speaker.extract_proposals(user, [], '')
            self.speaker.decide(Proposal(**proposals[0]), {}, [], [])

        prompts = [call['messages'][1]['content'] for call in calls]
        self.assertTrue(all('JSON' in prompt for prompt in prompts))
        self.assertIn('"proposals"', prompts[0])
        self.assertIn('"estimated_cost"', prompts[0])
        self.assertIn('"recommendation"', prompts[1])

    def test_bounded_candidate_repair_prompt_carries_current_meal_constraints(self):
        calls = []
        initial = [proposal("P1", cost=500), proposal("P2", mode="外卖", cost=500)]
        repaired = [proposal("P3", mode="外卖", cost=8), proposal("P4", mode="外卖", cost=12)]

        def fake_json(prompt):
            calls.append(prompt)
            return {"proposals": initial if len(calls) == 1 else repaired}

        user = UserInput(
            meal_budget=20,
            cooking_willingness="不愿意",
            shopping_willingness="不愿意",
            fridge="鸡蛋",
        )
        with patch.object(self.speaker, "_real_json", side_effect=fake_json):
            result = self.speaker.extract_proposals(user, [], "")
        self.assertEqual([proposal.id for proposal in result], ["P3", "P4"])
        self.assertEqual(len(calls), 2)
        self.assertIn('"meal_budget": 20.0', calls[1])
        self.assertIn('"cooking_willingness": "不愿意"', calls[1])
        self.assertIn("拒绝记录", calls[1])

    def test_mixed_imminent_expired_inventory_and_unexpired_inventory(self):
        candidates = [proposal('P1'), proposal('P2', mode='外卖'), proposal('P3', mode='外卖')]
        result = self.extract(UserInput(fridge='鸡蛋3天后过期、牛奶已过期'), candidates)
        self.assertFalse(result[0].feasible)
        result = self.extract(UserInput(fridge='鸡蛋3天后过期、牛奶未过期'), candidates)
        self.assertTrue(all(p.feasible for p in result))

    def test_no_invented_low_budget_fallback(self):
        candidates = [proposal(f'P{i}', cost=500) for i in range(1, 5)]
        with self.assertRaises(AgentError):
            self.extract(UserInput(budget=20, weekly_spend=19), candidates)

    def test_duplicate_adjustments_rejected_and_application_is_atomic(self):
        weights = calculate_weights(UserInput())
        original = copy.deepcopy(weights)
        adjustments = [{'agent': '钱包', 'delta': .25, 'reason': '证据'}] * 100
        with patch.object(self.speaker, '_real_json', return_value={'summary': 'test', 'adjustments': adjustments}):
            with self.assertRaises(AgentError):
                self.speaker.summarize(UserInput(), [], weights)
        with self.assertRaises(ValueError):
            Parliament._apply_adjustments(weights, adjustments)
        self.assertEqual(weights, original)
        Parliament._apply_adjustments(weights, [adjustments[0]])
        self.assertEqual(weights['钱包']['value'], 1.25)

    def test_nonfinite_money_and_scores_rejected_at_model_boundary(self):
        for value in (float('inf'), float('-inf'), float('nan')):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    Proposal(**proposal('P1', cost=value))
                with self.assertRaises(ValidationError):
                    UserInput(budget=value)
                with self.assertRaises(ValidationError):
                    UserInput(weekly_spend=value)
                with self.assertRaises(ValidationError):
                    Vote(agent='钱包', emoji='', round=1, scores={'P1': value}, rationale='')

    def test_votes_require_unique_roles_and_same_round(self):
        weights = calculate_weights(UserInput())
        proposals = [Proposal(**proposal('P1')), Proposal(**proposal('P2'))]
        votes = [Vote(agent=name, emoji='', round=1, scores={'P1': 10, 'P2': 1}, rationale='')
                 for name in weights]
        self.assertEqual(Parliament._aggregate(votes, weights, proposals), {'P1': 50, 'P2': 5})
        with self.assertRaises(ValueError):
            Parliament._aggregate([votes[0]] * 5, weights, proposals)
        with self.assertRaises(ValueError):
            Parliament._aggregate(votes[:-1], weights, proposals)
        votes[-1].round = 2
        with self.assertRaises(ValueError):
            Parliament._aggregate(votes, weights, proposals)

    def test_final_decision_locks_all_fact_fields(self):
        winner = Proposal(**proposal('P1'))
        hostile = {'dish': '红烧肉', 'steps': ['加入已过期鸡胸肉'], 'estimated_cost': 999}
        with patch.object(self.speaker, '_real_json', return_value=hostile):
            result = self.speaker.decide(winner, {}, [], [])
        for name in ('mode', 'dish', 'reason', 'estimated_cost', 'nutrition', 'steps'):
            self.assertEqual(getattr(result, name), getattr(winner, name))

    def test_specialist_and_public_core_imports_work_in_fresh_process(self):
        completed = subprocess.run([sys.executable, '-c',
                                    'from agents.speaker import Speaker; from core import Parliament; assert Speaker and Parliament'],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_mock_inventory_requires_every_recipe_ingredient(self):
        partial = proposal_seed(UserInput(fridge='只有西兰花'))[0]
        self.assertFalse(partial.feasible)
        self.assertIn('鸡胸肉', partial.reason)
        vegan_partial = proposal_seed(UserInput(fridge='番茄', special_needs='素食'))[0]
        self.assertFalse(vegan_partial.feasible)
        self.assertIn('豆腐', vegan_partial.reason)

    def test_fridge_debate_stays_with_current_inventory(self):
        agent = Fridge(mode='mock')
        other = type('MessageStub', (), {'id': 'M1', 'text': '我建议先看库存'})()
        self.assertIn('冰箱是空的', agent.mock_debate(UserInput(fridge='空'), other))
        self.assertIn('鸡胸、西兰花、糙米都在', agent.mock_debate(UserInput(fridge='鸡胸肉、西兰花、糙米'), other))


if __name__ == '__main__':
    unittest.main()
