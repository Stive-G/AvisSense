"""Optional Gradio interface backed by the same model as the FastAPI service."""

from functools import lru_cache

import gradio as gr

from src.inference import SentimentAnalyzer
from src.utils import clean_text, get_interpretation


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentAnalyzer:
    """Load the model once when the first Gradio prediction is requested."""
    return SentimentAnalyzer().load()


def predict_sentiment(text: str) -> str:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return "Veuillez saisir un avis a analyser."

    try:
        prediction = get_analyzer().predict(cleaned_text)
    except OSError as error:
        return f"Modele indisponible : {error}"
    return get_interpretation(prediction["label"], prediction["confidence"])


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Avis en francais"),
    outputs=gr.Textbox(label="Resultat"),
    title="AvisSense",
    description="Analyse de sentiment avec DistilCamemBERT fine-tune sur Allocine.",
    examples=[
        ["Un film magnifique, porte par des acteurs excellents."],
        ["Scenario previsible et mise en scene sans interet."],
    ],
)


if __name__ == "__main__":
    demo.launch()
