import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.nutritionist import Nutritionist
from core import llm_config


class LLMConfigTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.project_env = root / "project.env"
        self.local_env = root / "local.env"
        self.paths = patch.object(llm_config, "PROJECT_ENV_PATH", self.project_env), patch.object(llm_config, "LOCAL_ENV_PATH", self.local_env)
        for item in self.paths:
            item.start()
            self.addCleanup(item.stop)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.tempdir.cleanup)

    def test_precedence_is_process_then_local_then_project_and_reloadable(self):
        self.project_env.write_text("LLM_PROVIDER=openai\nOPENAI_MODEL=project-model\nOPENAI_API_KEY=project-key\n", encoding="utf-8")
        self.local_env.write_text("LLM_PROVIDER=deepseek\nDEEPSEEK_MODEL=local-model\nDEEPSEEK_API_KEY=local-key\n", encoding="utf-8")
        config = llm_config.get_llm_config()
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "local-model")
        self.assertTrue(config.configured)
        self.local_env.write_text("LLM_PROVIDER=deepseek\nDEEPSEEK_MODEL=reloaded-model\nDEEPSEEK_API_KEY=local-key\n", encoding="utf-8")
        self.assertEqual(llm_config.get_llm_config().model, "reloaded-model")
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "process-model", "OPENAI_API_KEY": "process-key"}):
            config = llm_config.get_llm_config()
        self.assertEqual(config.display_name, "openai/process-model")

    def test_deepseek_uses_fixed_endpoint_and_disables_thinking(self):
        self.local_env.write_text("LLM_PROVIDER=deepseek\nDEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
        config = llm_config.get_llm_config()
        self.assertEqual(config.model, "deepseek-flash")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.request_options, {"extra_body": {"thinking": {"type": "disabled"}}})
        self.assertNotIn("test-key", repr(config))

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            client = llm_config.create_client(config)
        self.assertEqual(client.kwargs["base_url"], "https://api.deepseek.com")
        self.assertEqual(client.kwargs["timeout"], 60.0)
        self.assertEqual(client.kwargs["max_retries"], 2)

    def test_blank_local_key_does_not_fall_back_to_project_key(self):
        self.project_env.write_text("LLM_PROVIDER=deepseek\nDEEPSEEK_API_KEY=project-key\n", encoding="utf-8")
        self.local_env.write_text("LLM_PROVIDER=deepseek\nDEEPSEEK_API_KEY=\n", encoding="utf-8")
        self.assertFalse(llm_config.get_llm_config().configured)

    def test_connection_check_is_explicit_and_bounded(self):
        self.local_env.write_text("LLM_PROVIDER=deepseek\nDEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="已连接"))])

        class FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeClient)}):
            self.assertEqual(llm_config.check_connection(), "deepseek-flash")
        self.assertEqual(calls[0]["max_tokens"], 64)
        self.assertEqual(calls[0]["timeout"], 60.0)
        self.assertEqual(calls[0]["extra_body"]["thinking"]["type"], "disabled")

    def test_real_mode_without_key_does_not_silently_mock(self):
        self.local_env.write_text("LLM_PROVIDER=deepseek\n", encoding="utf-8")
        agent = Nutritionist(mode="real")
        with self.assertRaisesRegex(Exception, "未配置.*API key"):
            agent.speak("", [], __import__("core.models", fromlist=["UserInput"]).UserInput())


if __name__ == "__main__":
    unittest.main()
