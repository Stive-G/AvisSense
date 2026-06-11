"""
push_model_to_hub.py — Envoie le modèle fine-tuné sur le Hugging Face Hub.

Pourquoi ? Le modèle (~270 Mo) est trop lourd pour GitHub. On le stocke sur
le Hub, et le Space déployé le télécharge automatiquement au démarrage
(via la variable d'environnement MODEL_ID de l'API).

Prérequis :
    1. Créer un compte sur https://huggingface.co
    2. Créer un token d'accès (Settings > Access Tokens > type "Write")
    3. Se connecter :  huggingface-cli login

Exécution (depuis la racine du projet) :
    python scripts/push_model_to_hub.py --repo VOTRE_PSEUDO/avissense-distilcamembert
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "sentiment_model"


def main():
    parser = argparse.ArgumentParser(description="Pousse le modèle fine-tuné sur le HF Hub")
    parser.add_argument("--repo", required=True,
                        help="Nom du repo Hub, ex : rima/avissense-distilcamembert")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        sys.exit(f"Erreur : modèle introuvable dans {MODEL_DIR}. Lancez d'abord train.py.")

    print(f"Chargement du modèle local depuis {MODEL_DIR} ...")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"Upload vers https://huggingface.co/{args.repo} ...")
    model.push_to_hub(args.repo)
    tokenizer.push_to_hub(args.repo)
    print("Terminé ! Le modèle est en ligne sur le Hub.")


if __name__ == "__main__":
    main()
