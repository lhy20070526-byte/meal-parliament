import unittest
from pathlib import Path
from unittest.mock import patch

from ui.hunger import hunger_slider


class HungerSliderTests(unittest.TestCase):
    def test_component_uses_streamlit_v1_post_message_bridge(self):
        html = Path(__file__).parents[1].joinpath("ui", "hunger_component", "index.html").read_text()
        self.assertNotIn("window.parent.Streamlit", html)
        self.assertIn('"streamlit:componentReady"', html)
        self.assertIn('"streamlit:setFrameHeight"', html)
        self.assertIn('"streamlit:setComponentValue"', html)
        self.assertIn('dataType: "json"', html)
        self.assertIn('"streamlit:render"', html)
        self.assertIn("event.source === window.parent", html)
        self.assertIn("Math.max(180", html)
        self.assertIn("ResizeObserver", html)
        self.assertIn("lastCommitted", html)
        self.assertIn("pointercancel", html)
        self.assertIn("range.disabled = Boolean(event.disabled)", html)

    def test_external_value_is_clamped_before_render(self):
        calls = []

        def component(**kwargs):
            calls.append(kwargs)
            return None

        with patch("ui.hunger._COMPONENT", side_effect=component):
            self.assertEqual(hunger_slider(99, key="hunger"), 10)
        self.assertEqual(calls[0]["value"], 10)
        self.assertEqual(calls[0]["default"], 10)
        self.assertEqual(calls[0]["key"], "hunger")

    def test_component_value_is_clamped_on_return(self):
        with patch("ui.hunger._COMPONENT", return_value=0):
            self.assertEqual(hunger_slider(5, key="hunger"), 1)
        with patch("ui.hunger._COMPONENT", return_value=12):
            self.assertEqual(hunger_slider(5, key="hunger"), 10)

    def test_invalid_values_fall_back_to_middle(self):
        with patch("ui.hunger._COMPONENT", return_value=None):
            self.assertEqual(hunger_slider("not-a-number", key="hunger"), 5)


if __name__ == "__main__":
    unittest.main()
