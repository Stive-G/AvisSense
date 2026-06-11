"""Interface Gradio minimale pour AvisSense."""

import gradio as gr


def predict_sentiment(text: str) -> str:
    """Retourne une prédiction temporaire en attendant le modèle final."""
    if not text or not text.strip():
        return "Veuillez saisir un avis à analyser."

    return "Modèle non encore chargé - structure prête."


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Avis en français"),
    outputs=gr.Textbox(label="Résultat"),
    title="AvisSense",
    description="Analyse intelligente du sentiment d'avis en français.",
)


if __name__ == "__main__":
    demo.launch()
