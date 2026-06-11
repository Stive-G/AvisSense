"""Small presentation helpers shared by user interfaces."""

import re


def clean_text(text: str | None) -> str:
    """Strip a user review and collapse repeated whitespace."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def format_label(label: str) -> str:
    """Return a consistent French display label."""
    labels = {
        "positive": "Positif",
        "positif": "Positif",
        "pos": "Positif",
        "negative": "Negatif",
        "negatif": "Negatif",
        "négatif": "Negatif",
        "neg": "Negatif",
    }
    return labels.get(str(label).lower(), str(label).capitalize())


def format_confidence(score: float) -> str:
    """Format a confidence score constrained to the [0, 1] interval."""
    score = max(0.0, min(1.0, float(score)))
    return f"{score:.1%}"


def get_interpretation(label: str, confidence: float) -> str:
    """Build the short result displayed by the Gradio interface."""
    return (
        f"Sentiment {format_label(label)} avec une confiance de "
        f"{format_confidence(confidence)}."
    )
