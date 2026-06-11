"""Tests for shared inference and presentation helpers."""

import unittest
from types import SimpleNamespace

import torch

from src.inference import SentimentAnalyzer
from src.utils import clean_text, get_interpretation


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return FakeBatch(input_ids=torch.tensor([[1, 2]]))

    def convert_ids_to_tokens(self, ids):
        return [f"token-{item}" for item in ids]


class FakeModel:
    config = SimpleNamespace(id2label={0: "negatif", 1: "positif"})

    def __call__(self, **inputs):
        return SimpleNamespace(logits=torch.tensor([[0.1, 2.0]]))


class SentimentAnalyzerTests(unittest.TestCase):
    def test_prediction_and_debug_details(self):
        analyzer = SentimentAnalyzer(device="cpu")
        analyzer.tokenizer = FakeTokenizer()
        analyzer.model = FakeModel()

        result = analyzer.predict("Excellent film", include_debug=True)

        self.assertEqual(result["label"], "positif")
        self.assertGreater(result["confidence"], 0.8)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=3)
        self.assertEqual(result["debug"]["tokens"], ["token-1", "token-2"])

    def test_text_helpers(self):
        self.assertEqual(clean_text("  tres   bon\nfilm  "), "tres bon film")
        self.assertIn("91.0%", get_interpretation("positif", 0.91))


if __name__ == "__main__":
    unittest.main()
