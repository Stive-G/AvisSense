"""Contract tests for the FastAPI endpoints without loading model weights."""

import unittest

from fastapi.testclient import TestClient

import api.main as api


class FakeAnalyzer:
    model_id = "fake/model"
    device = "cpu"
    is_loaded = True

    def predict(self, text):
        return {
            "label": "positif",
            "confidence": 0.9,
            "probabilities": {"negatif": 0.1, "positif": 0.9},
        }


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api.analyzer = FakeAnalyzer()
        cls.client = TestClient(api.app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_loaded"])

    def test_prediction(self):
        response = self.client.post("/predict", json={"text": "  Super   film  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["label"], "positif")

    def test_invalid_texts(self):
        self.assertEqual(self.client.post("/predict", json={"text": " "}).status_code, 400)
        self.assertEqual(
            self.client.post("/predict", json={"text": "x" * 5001}).status_code,
            400,
        )
        self.assertEqual(self.client.post("/predict", json={}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
