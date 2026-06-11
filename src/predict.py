"""Squelette de prédiction pour AvisSense."""

import argparse


def predict(text: str) -> str:
    """Prédit le sentiment d'un texte en français."""
    # TODO: Charger le tokenizer et le modèle fine-tuné.
    if not text or not text.strip():
        return "Texte vide."

    return "Modèle non encore chargé - prédiction temporaire."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prédire le sentiment d'un avis en français.")
    parser.add_argument("text", type=str, help="Texte de l'avis à analyser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict(args.text)
    print(result)


if __name__ == "__main__":
    main()
