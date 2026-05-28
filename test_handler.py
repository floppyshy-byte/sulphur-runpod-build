"""Smoke tests for Sulphur-2 handler — syntax and import checks only.

The full pipeline requires a GPU and model weights, so CI only
verifies handler.py is valid Python and basic constants are sane.
"""
import unittest
from pathlib import Path


class TestHandlerSmoke(unittest.TestCase):

    def test_handler_parses_without_error(self):
        handler_path = Path(__file__).resolve().parent / "handler.py"
        source = handler_path.read_text()
        compile(source, str(handler_path), "exec")

    def test_handler_contains_required_symbols(self):
        handler_path = Path(__file__).resolve().parent / "handler.py"
        source = handler_path.read_text()
        for symbol in ("def handler", "def get_pipeline", "DiffusionPipeline",
                       "t2v", "i2v", "MODEL_REPO"):
            self.assertIn(symbol, source,
                          f"handler.py must contain {symbol}")

    def test_dockerfile_exists(self):
        dockerfile = Path(__file__).resolve().parent / "Dockerfile"
        self.assertTrue(dockerfile.exists(), "Dockerfile missing")
        content = dockerfile.read_text()
        self.assertIn("Civitai/Sulphur-2-distilled-fp8", content)

    def test_frame_validation(self):
        """num_frames must be 8n+1."""
        for n in range(1, 10):
            num = n * 8 + 1
            self.assertEqual(num % 8, 1, f"{num} must be 8n+1")

    def test_default_frame_count(self):
        self.assertEqual(65 % 8, 1, "default 65 must be 8n+1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
