"""Évalue le modèle sauvegardé sur le test set Allociné."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "sentiment_model"
MAX_LENGTH = 256
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation du modèle sur le test set")
    parser.add_argument(
        "--max-test",
        type=int,
        default=2000,
        help="Nombre d'avis du test set à évaluer (max 20 000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Nombre d'avis traités par lot pendant l'évaluation",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=5,
        help="Nombre d'erreurs les plus confiantes à afficher",
    )
    return parser.parse_args()


def predict_in_batches(texts, model, tokenizer, batch_size, device):
    """Prédit une liste de textes par lots."""
    all_predictions = []
    all_confidences = []

    model.eval()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        # Le padding est fait au niveau du lot pour limiter le calcul inutile.
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probabilities, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

        done = min(start + batch_size, len(texts))
        print(f"\r  Progression : {done}/{len(texts)} avis", end="", flush=True)

    print()
    return np.array(all_predictions), np.array(all_confidences)


def show_most_confident_errors(texts, labels, predictions, confidences, id2label, n_errors):
    """Affiche les erreurs avec la confiance la plus élevée."""
    error_indices = np.where(predictions != labels)[0]
    if len(error_indices) == 0:
        print("\nAucune erreur sur cet échantillon.")
        return

    # Les erreurs les plus confiantes sont les plus utiles pour l'analyse qualitative.
    sorted_errors = error_indices[np.argsort(-confidences[error_indices])]

    print(f"\nTop {n_errors} erreurs les plus confiantes ({len(error_indices)} erreurs au total) :")
    print("-" * 70)
    for index in sorted_errors[:n_errors]:
        true_label = id2label[int(labels[index])]
        predicted_label = id2label[int(predictions[index])]
        print(
            f"  Vrai : {true_label:<8} | Prédit : {predicted_label:<8} "
            f"| Confiance : {confidences[index]:.2%}"
        )
        print(f"  « {texts[index][:160]}... »")
        print("-" * 70)


def main():
    args = parse_args()

    if not MODEL_DIR.exists():
        sys.exit(f"Erreur : modèle introuvable dans {MODEL_DIR}. Lancez d'abord : python scripts/train.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Matériel : {device} | Modèle : {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    id2label = model.config.id2label

    print("Chargement du test set Allociné...")
    # Le test set reste séparé du train pour garder une mesure finale honnête.
    test_data = load_dataset("allocine", split="test").shuffle(seed=SEED)
    test_data = test_data.select(range(min(args.max_test, len(test_data))))
    texts = test_data["review"]
    labels = np.array(test_data["label"])
    print(f"Évaluation sur {len(texts)} avis de test\n")

    predictions, confidences = predict_in_batches(
        texts, model, tokenizer, args.batch_size, device
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    print("\nMétriques sur le test set :")
    print(f"  accuracy  : {accuracy_score(labels, predictions):.4f}")
    print(f"  precision : {precision:.4f}")
    print(f"  recall    : {recall:.4f}")
    print(f"  f1        : {f1:.4f}")
    print(f"  confiance moyenne : {confidences.mean():.4f}")

    print("\nRapport par classe :")
    print(
        classification_report(
            labels,
            predictions,
            target_names=["négatif", "positif"],
            digits=4,
        )
    )

    matrix = confusion_matrix(labels, predictions)
    print("Matrice de confusion :")
    print(f"{'':>16} | {'prédit négatif':>15} | {'prédit positif':>15}")
    print("-" * 52)
    print(f"{'vrai négatif':>16} | {matrix[0][0]:>15} | {matrix[0][1]:>15}")
    print(f"{'vrai positif':>16} | {matrix[1][0]:>15} | {matrix[1][1]:>15}")

    show_most_confident_errors(
        texts, labels, predictions, confidences, id2label, args.show_errors
    )


if __name__ == "__main__":
    main()
