"""Interface Gradio optionnelle basée sur le même modèle que l'API."""

from functools import lru_cache

import gradio as gr

from src.inference import SentimentAnalyzer
from src.utils import clean_text, get_interpretation


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentAnalyzer:
    """Charge le modèle une seule fois pour Gradio."""
    return SentimentAnalyzer().load()


def predict_sentiment(text: str) -> str:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return "Veuillez saisir un avis à analyser."

    try:
        prediction = get_analyzer().predict(cleaned_text)
    except OSError as error:
        return f"Modèle indisponible : {error}"
    return get_interpretation(prediction["label"], prediction["confidence"])


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Avis en français"),
    outputs=gr.Textbox(label="Résultat"),
    title="AvisSense",
    description="Analyse de sentiment avec DistilCamemBERT fine-tuné sur Allociné.",
    examples=[
        ["Un film magnifique, porté par des acteurs excellents."],
        ["Scénario prévisible et mise en scène sans intérêt."],
    ],
)


if __name__ == "__main__":
    demo.launch()
