"""Fonctions utilitaires partagées par les interfaces du projet."""

import re
import unicodedata


def clean_text(text: str | None) -> str:
    """Nettoie un texte utilisateur et compacte les espaces."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_label(label: str) -> str:
    # Normalise les accents pour comparer proprement des labels venant de sources différentes.
    normalized = unicodedata.normalize("NFD", str(label))
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return normalized.lower()


def format_label(label: str) -> str:
    """Retourne une étiquette française cohérente pour l'affichage."""
    labels = {
        "positive": "Positif",
        "positif": "Positif",
        "pos": "Positif",
        "negative": "Négatif",
        "negatif": "Négatif",
        "neg": "Négatif",
    }
    return labels.get(_normalize_label(label), str(label).capitalize())


def format_confidence(score: float) -> str:
    """Formate une confiance dans l'intervalle [0, 1]."""
    score = max(0.0, min(1.0, float(score)))
    return f"{score:.1%}"


def get_interpretation(label: str, confidence: float) -> str:
    """Construit un résultat court lisible par l'utilisateur."""
    return f"Sentiment {format_label(label)} avec une confiance de {format_confidence(confidence)}."
