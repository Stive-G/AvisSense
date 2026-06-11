"""Fonctions utilitaires simples pour AvisSense."""

import re


def clean_text(text: str) -> str:
    """Nettoie légèrement un texte utilisateur."""
    if text is None:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_label(label: str) -> str:
    """Normalise l'affichage d'un label de sentiment."""
    labels = {
        "positive": "Positif",
        "negative": "Négatif",
        "neutral": "Neutre",
        "pos": "Positif",
        "neg": "Négatif",
    }
    return labels.get(str(label).lower(), str(label).capitalize())


def format_confidence(score: float) -> str:
    """Formate un score de confiance entre 0 et 1."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "Confiance inconnue"

    score = max(0.0, min(1.0, score))
    return f"{score:.1%}"


def get_interpretation(label: str, confidence: float) -> str:
    """Retourne une interprétation courte du résultat."""
    formatted_label = format_label(label)
    formatted_confidence = format_confidence(confidence)
    return f"Sentiment {formatted_label} avec une confiance de {formatted_confidence}."
