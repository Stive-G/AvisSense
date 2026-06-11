"""Publie le modèle fine-tuné sur le Hugging Face Hub."""

import argparse
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "sentiment_model"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publie le modèle fine-tuné sur le Hugging Face Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help="Nom du dépôt Hub, par exemple : utilisateur/avissense-distilcamembert",
    )
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        sys.exit(f"Erreur : modèle introuvable dans {MODEL_DIR}. Lancez d'abord train.py.")

    # Le tokenizer doit être publié avec les poids pour reproduire l'inférence.
    print(f"Chargement du modèle local depuis {MODEL_DIR} ...")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"Envoi vers https://huggingface.co/{args.repo} ...")
    model.push_to_hub(args.repo)
    tokenizer.push_to_hub(args.repo)
    print("Publication terminée.")
    print(f"Variable à utiliser sur le Space : MODEL_ID = {args.repo}")


if __name__ == "__main__":
    main()
