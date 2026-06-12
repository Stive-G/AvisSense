"""Génère et sauvegarde les graphiques de métriques du modèle (PNG).

Produit les figures classiques d'un rapport de projet ML, prêtes à insérer
dans les slides ou la vidéo :

    reports/figures/
    ├── 1_loss_curves.png            Courbes de perte train/validation (diagnostic surapprentissage)
    ├── 2_metrics_par_epoque.png     Accuracy et F1 de validation par époque
    ├── 3_confusion_matrix_abs.png   Matrice de confusion (valeurs absolues)
    ├── 4_confusion_matrix_pct.png   Matrice de confusion (% par classe réelle)
    └── 5_baseline_vs_finetune.png   Baseline (modèle brut) vs fine-tuné : gain du transfer learning

Prérequis :
    - un modèle entraîné dans model/sentiment_model (python scripts/train.py) ;
    - les figures 1 et 2 demandent training_history.json, écrit par train.py
      (relancer un entraînement si le fichier manque : anciens runs ne l'ont pas).

Exécution (depuis la racine du projet) :
    python scripts/generate_metrics_plots.py
    python scripts/generate_metrics_plots.py --max-test 2000 --skip-baseline
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Rendu hors écran : fonctionne sans interface graphique (serveur, CI)
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "sentiment_model"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
BASE_MODEL = "cmarkea/distilcamembert-base"  # Pour la baseline « brute » (non fine-tunée)
CLASS_NAMES = ["négatif", "positif"]
MAX_LENGTH = 256
SEED = 42

# Style commun à toutes les figures
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def parse_args():
    parser = argparse.ArgumentParser(description="Génère les graphiques de métriques en PNG")
    parser.add_argument("--max-test", type=int, default=2000,
                        help="Nombre d'avis du test set à évaluer")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Taille des lots pour l'inférence")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Saute la figure 5 (évite une 2e évaluation, plus rapide)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Figures 1 et 2 — à partir de l'historique d'entraînement
# ─────────────────────────────────────────────────────────────────────────────
def load_history():
    """Charge training_history.json s'il existe (écrit par train.py)."""
    history_path = MODEL_DIR / "training_history.json"
    if not history_path.exists():
        return None
    with open(history_path, encoding="utf-8") as file:
        return json.load(file)


def plot_loss_curves(history, output_path):
    """Courbes de perte : la training loss doit descendre ; si la validation
    loss remonte alors que la training loss descend, c'est du surapprentissage."""
    # log_history mélange plusieurs types d'entrées : on sépare les pas
    # d'entraînement (clé "loss") des évaluations par époque (clé "eval_loss").
    train_points = [(entry["epoch"], entry["loss"]) for entry in history if "loss" in entry]
    eval_points = [(entry["epoch"], entry["eval_loss"]) for entry in history if "eval_loss" in entry]

    figure, axis = plt.subplots(figsize=(6, 4))
    if train_points:
        axis.plot(*zip(*train_points), marker="o", markersize=3,
                  label="Training loss", color="#1f77b4")
    if eval_points:
        axis.plot(*zip(*eval_points), marker="s", markersize=5,
                  label="Validation loss", color="#ff7f0e")
    axis.set_xlabel("Époque")
    axis.set_ylabel("Loss")
    axis.set_title("Courbes de perte (diagnostic surapprentissage)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def plot_epoch_metrics(history, output_path):
    """Accuracy et F1 mesurés sur la validation à la fin de chaque époque."""
    epochs = [entry["epoch"] for entry in history if "eval_accuracy" in entry]
    accuracies = [entry["eval_accuracy"] for entry in history if "eval_accuracy" in entry]
    f1_scores = [entry["eval_f1"] for entry in history if "eval_f1" in entry]

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, accuracies, marker="o", label="Accuracy", color="#1f77b4")
    axis.plot(epochs, f1_scores, marker="s", label="F1", color="#ff7f0e")
    axis.set_xlabel("Époque")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Métriques de validation par époque")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation sur le test set (partagée par les figures 3, 4 et 5)
# ─────────────────────────────────────────────────────────────────────────────
def predict_in_batches(texts, model, tokenizer, batch_size, device):
    """Prédit une liste de textes par lots (même logique que evaluate.py)."""
    predictions = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        done = min(start + batch_size, len(texts))
        print(f"\r  Progression : {done}/{len(texts)} avis", end="", flush=True)
    print()
    return np.array(predictions)


def plot_confusion(matrix, output_path, as_percent=False):
    """Matrice de confusion en heatmap, valeurs absolues ou % par classe réelle."""
    if as_percent:
        # Normalisation LIGNE par ligne : chaque ligne (classe réelle) somme à 100 %.
        display = matrix / matrix.sum(axis=1, keepdims=True) * 100
        title = "Matrice de confusion — % par classe réelle"
        text_format = "{:.1f}"
    else:
        display = matrix.astype(float)
        title = "Matrice de confusion — valeurs absolues"
        text_format = "{:.0f}"

    figure, axis = plt.subplots(figsize=(5, 4.2))
    image = axis.imshow(display, cmap="Blues")
    axis.set_xticks(range(len(CLASS_NAMES)), [f"prédit {name}" for name in CLASS_NAMES])
    axis.set_yticks(range(len(CLASS_NAMES)), [f"réel {name}" for name in CLASS_NAMES])
    axis.set_title(title)
    axis.grid(False)
    # Annotation de chaque case ; texte blanc sur les cases foncées
    threshold = display.max() / 2
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            color = "white" if display[row][column] > threshold else "black"
            axis.text(column, row, text_format.format(display[row][column]),
                      ha="center", va="center", color=color, fontsize=12)
    figure.colorbar(image, fraction=0.046)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def plot_baseline_comparison(baseline_scores, finetuned_scores, output_path):
    """Barres comparées : modèle brut (tête aléatoire) vs modèle fine-tuné.

    C'est la figure qui montre le gain apporté par le fine-tuning : la
    baseline a le même backbone mais une tête non entraînée (~50 % = hasard
    sur 2 classes équilibrées).
    """
    metric_names = ["Accuracy", "F1"]
    positions = np.arange(len(metric_names))
    bar_width = 0.35

    figure, axis = plt.subplots(figsize=(6, 4))
    bars_baseline = axis.bar(positions - bar_width / 2, baseline_scores, bar_width,
                             label="Baseline (brut)", color="#c8c8c8")
    bars_finetuned = axis.bar(positions + bar_width / 2, finetuned_scores, bar_width,
                              label="Fine-tuné", color="#1f77b4")
    # Valeur affichée au-dessus de chaque barre
    for bars in (bars_baseline, bars_finetuned):
        axis.bar_label(bars, fmt="%.3f", padding=2)
    axis.set_xticks(positions, metric_names)
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Baseline vs fine-tuné — gain apporté par le transfer learning")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main():
    args = parse_args()

    if not MODEL_DIR.exists():
        sys.exit(f"Erreur : modèle introuvable dans {MODEL_DIR}. "
                 "Lancez d'abord : python scripts/train.py")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Figures 1 & 2 : historique d'entraînement ───────────────────────────
    history = load_history()
    if history:
        plot_loss_curves(history, FIGURES_DIR / "1_loss_curves.png")
        plot_epoch_metrics(history, FIGURES_DIR / "2_metrics_par_epoque.png")
        print("Figures 1-2 (historique d'entraînement) : OK")
    else:
        print("training_history.json absent : figures 1-2 sautées.\n"
              "(Relancez python scripts/train.py — le fichier est écrit depuis "
              "la mise à jour du script ; les anciens runs ne l'ont pas.)")

    # ── Test set partagé par les figures 3, 4 et 5 ──────────────────────────
    print("\nChargement du test set Allociné...")
    test_data = load_dataset("allocine", split="test").shuffle(seed=SEED)
    test_data = test_data.select(range(min(args.max_test, len(test_data))))
    texts = test_data["review"]
    labels = np.array(test_data["label"])

    # ── Figures 3 & 4 : matrices de confusion du modèle fine-tuné ───────────
    print(f"\nÉvaluation du modèle fine-tuné sur {len(texts)} avis :")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    predictions = predict_in_batches(texts, model, tokenizer, args.batch_size, device)

    matrix = confusion_matrix(labels, predictions)
    plot_confusion(matrix, FIGURES_DIR / "3_confusion_matrix_abs.png", as_percent=False)
    plot_confusion(matrix, FIGURES_DIR / "4_confusion_matrix_pct.png", as_percent=True)
    finetuned_scores = [accuracy_score(labels, predictions),
                        f1_score(labels, predictions)]
    print(f"Figures 3-4 (confusion) : OK — accuracy={finetuned_scores[0]:.4f}, "
          f"f1={finetuned_scores[1]:.4f}")

    # ── Figure 5 : baseline brute vs fine-tuné ──────────────────────────────
    if args.skip_baseline:
        print("Figure 5 sautée (--skip-baseline).")
    else:
        # Le warning « weights newly initialized » est NORMAL : c'est justement
        # le principe de la baseline — la tête de classification n'est pas entraînée.
        print(f"\nÉvaluation de la baseline brute ({BASE_MODEL}) :")
        baseline_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        baseline_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2
        ).to(device)
        baseline_predictions = predict_in_batches(
            texts, baseline_model, baseline_tokenizer, args.batch_size, device
        )
        baseline_scores = [accuracy_score(labels, baseline_predictions),
                           f1_score(labels, baseline_predictions)]
        plot_baseline_comparison(baseline_scores, finetuned_scores,
                                 FIGURES_DIR / "5_baseline_vs_finetune.png")
        print(f"Figure 5 (baseline) : OK — baseline accuracy={baseline_scores[0]:.4f} "
              f"vs fine-tuné {finetuned_scores[0]:.4f}")

    print(f"\nToutes les figures sont dans : {FIGURES_DIR}")


if __name__ == "__main__":
    main()
