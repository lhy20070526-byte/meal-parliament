import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.llm_config import LLMConfig
from core.needs import NeedsError, parse_needs


class NeedsTests(unittest.TestCase):
    def test_without_service_requires_manual_entry(self):
        config = LLMConfig(provider="deepseek", model="deepseek-flash", parliament_mode="auto")
        with patch("core.needs.get_llm_config", return_value=config), patch("core.needs.create_client") as create:
            with self.assertRaisesRegex(NeedsError, "手动填写"):
                parse_needs("这顿最多花30元")
        create.assert_not_called()

    def test_length_is_limited_before_service_call(self):
        with self.assertRaisesRegex(NeedsError, "4000"):
            parse_needs("字" * 4001)

    def test_only_explicit_fields_are_returned_and_payload_is_json(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                    "meal_budget": 30,
                    "shopping_willingness": "不愿意",
                    "cooking_willingness": "愿意",
                    "fridge": "鸡蛋",
                    "taste": "清淡",
                    "hunger": 8,
                    "time": "19:00",
                    "special_needs": "高血压",
                    "made_up": "不要采用",
                }, ensure_ascii=False)))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        config = LLMConfig(provider="deepseek", model="deepseek-flash", api_key="local-test", parliament_mode="real")
        with patch("core.needs.get_llm_config", return_value=config), patch("core.needs.create_client", return_value=client):
            result = parse_needs("这顿最多花30元，不愿意买菜，愿意做饭。冰箱有鸡蛋，想吃清淡。")
        self.assertEqual(result, {
            "meal_budget": 30.0,
            "shopping_willingness": "不愿意",
            "cooking_willingness": "愿意",
            "fridge": "鸡蛋",
            "taste": "清淡",
        })
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertIn("JSON", calls[0]["messages"][1]["content"])

    def test_invalid_model_json_is_safe_chinese_error(self):
        class Completions:
            def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="不是json"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        config = LLMConfig(provider="deepseek", model="deepseek-flash", api_key="local-test", parliament_mode="real")
        with patch("core.needs.get_llm_config", return_value=config), patch("core.needs.create_client", return_value=client):
            with self.assertRaisesRegex(NeedsError, "没有完成"):
                parse_needs("我想吃清淡的")

    def test_numeric_fields_require_a_number_in_the_original_note(self):
        class Completions:
            def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                    "time": "23:00", "meal_budget": 500, "hunger": 8,
                })))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        config = LLMConfig(provider="deepseek", model="deepseek-flash", api_key="local-test", parliament_mode="real")
        with patch("core.needs.get_llm_config", return_value=config), patch("core.needs.create_client", return_value=client):
            result = parse_needs("预算还没想好，今晚想吃番茄面")
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
