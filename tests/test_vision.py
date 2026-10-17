import base64
import json
import unittest
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from core.llm_config import LLMConfig
from core.vision import VisionError, prepare_image, recognize_fridge_image


def image_bytes(*, fmt="PNG", size=(40, 20), exif=False):
    image = Image.new("RGB", size, "white")
    output = BytesIO()
    if exif:
        image.save(output, format=fmt, exif=b"Exif\x00\x00fake")
    else:
        image.save(output, format=fmt)
    return output.getvalue()


class VisionTests(unittest.TestCase):
    def test_prepare_image_reencodes_and_limits_side_without_exif(self):
        prepared, mime = prepare_image(image_bytes(fmt="JPEG", size=(2400, 1000), exif=True))
        self.assertEqual(mime, "image/jpeg")
        with Image.open(BytesIO(prepared)) as image:
            self.assertEqual(max(image.size), 1600)
            self.assertFalse(image.getexif())

    def test_mock_mode_does_not_send_image_and_is_explicit(self):
        called = []
        config = LLMConfig(provider="deepseek", model="deepseek-flash", parliament_mode="mock")
        result = recognize_fridge_image(image_bytes(), config=config, client=lambda: called.append(True))
        self.assertEqual(result.status, "mock")
        self.assertIn("示例", result.notice)
        self.assertEqual(called, [])

    def test_real_payload_contains_data_image_and_json_contract(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                    "items": [{"name": "鸡蛋", "quantity": "3个", "confidence": 0.9}],
                    "uncertainties": ["照片看不出新鲜程度"],
                }, ensure_ascii=False)))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        config = LLMConfig(provider="deepseek", model="deepseek-flash", api_key="test", parliament_mode="real")
        result = recognize_fridge_image(image_bytes(fmt="WEBP"), config=config, client=client)
        self.assertEqual(result.items[0].name, "鸡蛋")
        payload = calls[0]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        content = payload["messages"][1]["content"]
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        decoded = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
        self.assertTrue(decoded.startswith(b"\xff\xd8"))
        self.assertIn("JSON", content[0]["text"])

    def test_invalid_image_is_rejected_before_model_call(self):
        config = LLMConfig(provider="deepseek", model="deepseek-flash", api_key="test", parliament_mode="real")
        with self.assertRaises(VisionError):
            recognize_fridge_image(b"not an image", config=config)


if __name__ == "__main__":
    unittest.main()
