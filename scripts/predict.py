"""
predict.py — Prédiction du sentiment d'un avis avec le modèle fine-tuné.
=========================================================================

Ce script fait l'inférence "à la main" (sans la pipeline Hugging Face)
pour montrer explicitement les 4 étapes d'une prédiction :

    1. Tokenisation : texte -> identifiants de tokens (tenseurs)
    2. Forward pass : le modèle produit 2 logits (scores bruts, un par classe)
    3. Softmax      : logits -> probabilités qui somment à 1
    4. Argmax       : la classe avec la plus haute probabilité = la prédiction
                      sa probabilité = le score de confiance

Exécution (depuis la racine du projet) :
    # Un avis en argument :
    python scripts/predict.py "Ce film est un chef-d'œuvre absolu !"

    # Mode interactif (taper des avis l'un après l'autre, 'q' pour quitter) :
    python scripts/predict.py
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Chemin du modèle fine-tuné (même convention que train.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "sentiment_model"

# Doit correspondre à la longueur utilisée à l'entraînement
MAX_LENGTH = 256


def load_model_and_tokenizer():
    """Charge le modèle fine-tuné et son tokenizer depuis le disque.

    Les deux sont indissociables : le tokenizer convertit le texte en
    identifiants qui correspondent exactement aux embeddings du modèle.
    """
    if not MODEL_DIR.exists():
        sys.exit(
            f"Erreur : modèle introuvable dans {MODEL_DIR}\n"
            "Lancez d'abord l'entraînement : python scripts/train.py"
        )

    print(f"Chargement du modèle depuis {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    # Mode évaluation : désactive le dropout (couches aléatoires utilisées
    # uniquement pendant l'entraînement) -> prédictions déterministes.
    model.eval()
    return model, tokenizer


def predict_sentiment(text: str, model, tokenizer, verbose: bool = False) -> dict:
    """Prédit le sentiment d'un texte. Renvoie label + confiance + détail.

    Chaque étape de l'inférence est explicite (voir docstring du module).
    """
    # --- Étape 1 : tokenisation -------------------------------------------
    # return_tensors="pt" -> tenseurs PyTorch directement utilisables.
    # truncation=True     -> coupe à 256 tokens si l'avis est trop long.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    if verbose:
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"  Tokens ({len(tokens)}) : {tokens[:15]}{'...' if len(tokens) > 15 else ''}")

    # --- Étape 2 : forward pass --------------------------------------------
    # torch.no_grad() : pas de calcul de gradients (on n'entraîne pas),
    # c'est plus rapide et consomme moins de mémoire.
    with torch.no_grad():
        outputs = model(**inputs)

    # logits : scores bruts, shape (1, 2) -> [score_négatif, score_positif].
    # Ils ne sont PAS des probabilités (peuvent être négatifs, ne somment pas à 1).
    logits = outputs.logits[0]

    # --- Étape 3 : softmax ---------------------------------------------------
    # softmax(z_i) = exp(z_i) / somme(exp(z_j)) : transforme les logits en
    # probabilités positives qui somment à 1.
    probabilities = torch.softmax(logits, dim=-1)

    # --- Étape 4 : argmax + score de confiance ------------------------------
    predicted_class_id = int(torch.argmax(probabilities))
    confidence = float(probabilities[predicted_class_id])

    # id2label a été enregistré dans la config du modèle par train.py :
    # {0: "négatif", 1: "positif"}
    label = model.config.id2label[predicted_class_id]

    if verbose:
        print(f"  Logits bruts          : négatif={logits[0]:.3f}, positif={logits[1]:.3f}")
        print(f"  Probabilités (softmax): négatif={probabilities[0]:.4f}, "
              f"positif={probabilities[1]:.4f}")

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "négatif": round(float(probabilities[0]), 4),
            "positif": round(float(probabilities[1]), 4),
        },
    }


def print_prediction(text: str, prediction: dict):
    """Affichage formaté d'une prédiction."""
    emoji = "😊" if prediction["label"] == "positif" else "😞"
    print(f"\n  Avis      : {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"  Sentiment : {emoji} {prediction['label'].upper()}")
    print(f"  Confiance : {prediction['confidence']:.2%}")


def interactive_mode(model, tokenizer):
    """Boucle interactive : l'utilisateur tape des avis, 'q' pour quitter."""
    print("\nMode interactif — tapez un avis puis Entrée ('q' pour quitter)\n")
    while True:
        try:
            text = input("Votre avis > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            print("  (texte vide, réessayez)")
            continue
        prediction = predict_sentiment(text, model, tokenizer, verbose=True)
        print_prediction(text, prediction)
        print()
    print("Au revoir !")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse de sentiment d'un avis en français"
    )
    parser.add_argument("text", nargs="?", default=None,
                        help="L'avis à analyser (entre guillemets). "
                             "Sans argument : mode interactif.")
    parser.add_argument("--verbose", action="store_true",
                        help="Affiche les tokens, logits et probabilités détaillés")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()

    if args.text is None:
        # Aucun texte fourni -> mode interactif
        interactive_mode(model, tokenizer)
    else:
        text = args.text.strip()
        if not text:
            sys.exit("Erreur : le texte est vide.")
        prediction = predict_sentiment(text, model, tokenizer, verbose=args.verbose)
        print_prediction(text, prediction)


if __name__ == "__main__":
    main()
