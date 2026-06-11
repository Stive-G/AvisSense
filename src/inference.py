"""Chargement et inférence partagés entre API, CLI et Gradio."""

import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = str(PROJECT_ROOT / "model" / "sentiment_model")
MAX_LENGTH = 256


class SentimentAnalyzer:
    """Charge le modèle une fois et expose une méthode de prédiction réutilisable."""

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> "SentimentAnalyzer":
        if self.is_loaded:
            return self

        model_path = Path(self.model_id)
        # On vérifie le chemin local explicitement pour renvoyer une erreur claire.
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable dans {model_path}. Lancez d'abord scripts/train.py."
            )

        # Même interface pour un chemin local ou un repo Hugging Face via MODEL_ID.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()
        return self

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None

    def predict(self, text: str, include_debug: bool = False) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Le modèle doit être chargé avant la prédiction.")

        # La troncature garde le comportement aligné avec l'entraînement.
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]

        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = int(torch.argmax(probabilities))

        result = {
            "label": self.model.config.id2label[predicted_class_id],
            "confidence": round(float(probabilities[predicted_class_id]), 4),
            "probabilities": {
                self.model.config.id2label[i]: round(float(probability), 4)
                for i, probability in enumerate(probabilities)
            },
        }
        if include_debug:
            result["debug"] = {
                "tokens": self.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][0].detach().cpu().tolist()
                ),
                "logits": [round(float(logit), 4) for logit in logits],
            }
        return result
