"""
evaluate.py — Évaluation détaillée du modèle sauvegardé sur le test set Allociné.
==================================================================================

CE QUE FAIT CE SCRIPT (à lancer APRÈS train.py) :
    1. Recharge le modèle fine-tuné depuis model/sentiment_model/
    2. Prédit le sentiment des avis du split TEST (jamais vus à l'entraînement)
    3. Affiche : accuracy, precision, recall, F1, rapport par classe,
       matrice de confusion
    4. Montre les ERREURS LES PLUS CONFIANTES : les avis où le modèle se
       trompe en étant très sûr de lui. C'est l'analyse la plus instructive
       du projet : on y trouve l'ironie, les avis mitigés, le vocabulaire
       ambigu... -> matière directe pour la section "limites".

POURQUOI UN SCRIPT SÉPARÉ DE train.py ?
    - On peut ré-évaluer le modèle à tout moment sans ré-entraîner (20 min
      économisées à chaque fois).
    - On peut évaluer sur plus d'avis que pendant l'entraînement.
    - Séparation des responsabilités : entraîner et évaluer sont deux
      activités distinctes du cycle de vie d'un modèle.

COMMENT LANCER (depuis la racine du projet) :
    python scripts/evaluate.py
    python scripts/evaluate.py --max-test 5000 --show-errors 10
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import argparse            # Options en ligne de commande
import sys                 # Sortie propre en cas d'erreur
from pathlib import Path   # Chemins portables

import numpy as np         # Tableaux numériques (tri des erreurs, masques)
import torch               # Exécution du modèle
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,            # % de bonnes réponses
    classification_report,     # Rapport precision/recall/F1 PAR classe
    confusion_matrix,          # Où sont les erreurs
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─── CONSTANTES (mêmes conventions que train.py) ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "sentiment_model"
MAX_LENGTH = 256   # Même longueur max qu'à l'entraînement
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation du modèle sur le test set")
    parser.add_argument("--max-test", type=int, default=2000,
                        help="Nombre d'avis du test set à évaluer (max 20 000)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Nombre d'avis prédits en même temps. Plus grand = "
                             "plus rapide, mais plus de mémoire")
    parser.add_argument("--show-errors", type=int, default=5,
                        help="Nombre d'erreurs les plus confiantes à afficher")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Prédiction par lots (batch) — bien plus rapide qu'avis par avis
# ─────────────────────────────────────────────────────────────────────────────
def predict_in_batches(texts, model, tokenizer, batch_size, device):
    """Prédit le sentiment d'une liste de textes, par lots.

    POURQUOI PAR LOTS ? Prédire 2000 avis un par un = 2000 passages dans le
    modèle. Par lots de 32, on n'en fait que 63 : le GPU/CPU calcule les 32
    avis EN PARALLÈLE dans les mêmes opérations matricielles.

    Renvoie deux tableaux numpy alignés avec `texts` :
        predictions : la classe prédite (0 ou 1) pour chaque texte
        confidences : la probabilité softmax de la classe prédite
    """
    all_predictions = []
    all_confidences = []

    model.eval()   # Mode évaluation : dropout désactivé (cf. predict.py)

    # range(0, N, batch_size) découpe la liste en tranches de 32
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        # Tokenisation du lot entier d'un coup.
        # padding=True : les avis du lot n'ont pas la même longueur -> on
        # complète les courts avec des tokens <pad> jusqu'à la longueur du
        # plus long DU LOT (l'attention_mask dira au modèle de les ignorer).
        # .to(device) : envoie les tenseurs sur le GPU si disponible.
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(device)

        # Inférence sans calcul de gradients (on ne fait que prédire)
        with torch.no_grad():
            logits = model(**inputs).logits   # forme (batch_size, 2)

        # softmax ligne par ligne : chaque avis a ses 2 probabilités
        probabilities = torch.softmax(logits, dim=-1)
        # torch.max renvoie EN MÊME TEMPS la valeur max (= la confiance) et
        # son indice (= la classe prédite), pour chaque ligne du lot.
        confidences, predictions = torch.max(probabilities, dim=-1)

        # .cpu().numpy() : rapatrie les résultats du GPU vers des tableaux numpy
        all_predictions.extend(predictions.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

        # Barre de progression maison (le \r réécrit la même ligne)
        done = min(start + batch_size, len(texts))
        print(f"\r  Progression : {done}/{len(texts)} avis", end="", flush=True)

    print()  # Retour à la ligne final
    return np.array(all_predictions), np.array(all_confidences)


# ─────────────────────────────────────────────────────────────────────────────
# Analyse qualitative : les erreurs les plus confiantes
# ─────────────────────────────────────────────────────────────────────────────
def show_most_confident_errors(texts, labels, predictions, confidences,
                               id2label, n_errors):
    """Affiche les erreurs où le modèle était le plus sûr de lui.

    POURQUOI C'EST INTÉRESSANT ? Une erreur à 51 % de confiance = le modèle
    hésitait, c'est excusable. Une erreur à 99 % = le modèle est
    SYSTÉMATIQUEMENT trompé par quelque chose : ironie ("Bravo, 2h de
    perdues"), avis mitigé, négation complexe... Ce sont ces cas qu'on
    cite dans la section "limites" du projet.
    """
    # np.where renvoie les indices où la condition est vraie (les erreurs)
    error_indices = np.where(predictions != labels)[0]
    if len(error_indices) == 0:
        print("\nAucune erreur sur cet échantillon !")
        return

    # On trie les erreurs par confiance DÉCROISSANTE :
    # argsort trie en croissant, le signe - inverse l'ordre.
    sorted_errors = error_indices[np.argsort(-confidences[error_indices])]

    print(f"\nTop {n_errors} erreurs les plus confiantes "
          f"({len(error_indices)} erreurs au total) :")
    print("-" * 70)
    for index in sorted_errors[:n_errors]:
        true_label = id2label[int(labels[index])]
        predicted_label = id2label[int(predictions[index])]
        print(f"  Vrai : {true_label:<8} | Prédit : {predicted_label:<8} "
              f"| Confiance : {confidences[index]:.2%}")
        print(f"  « {texts[index][:160]}... »")
        print("-" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── 1. Recharger le modèle fine-tuné ────────────────────────────────────
    if not MODEL_DIR.exists():
        sys.exit(f"Erreur : modèle introuvable dans {MODEL_DIR}. "
                 "Lancez d'abord : python scripts/train.py")

    # "cuda" = GPU NVIDIA ; sinon on calcule sur le processeur
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Matériel : {device} | Modèle : {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    id2label = model.config.id2label   # {0: "négatif", 1: "positif"} (écrit par train.py)

    # ── 2. Charger le test set ──────────────────────────────────────────────
    # split="test" : on ne télécharge QUE le split de test.
    # Le test set n'a servi ni à entraîner ni à choisir le meilleur
    # checkpoint -> c'est la mesure honnête de la généralisation.
    print("Chargement du test set Allociné...")
    test_data = load_dataset("allocine", split="test").shuffle(seed=SEED)
    # min(...) : sécurité si on demande plus d'avis qu'il n'en existe
    test_data = test_data.select(range(min(args.max_test, len(test_data))))
    texts = test_data["review"]
    labels = np.array(test_data["label"])
    print(f"Évaluation sur {len(texts)} avis de test\n")

    # ── 3. Prédire tous les avis ────────────────────────────────────────────
    predictions, confidences = predict_in_batches(
        texts, model, tokenizer, args.batch_size, device
    )

    # ── 4. Métriques globales ───────────────────────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"   # la classe 1 (positif) = référence
    )
    print("\nMétriques sur le test set :")
    print(f"  accuracy  : {accuracy_score(labels, predictions):.4f}")
    print(f"  precision : {precision:.4f}")
    print(f"  recall    : {recall:.4f}")
    print(f"  f1        : {f1:.4f}")
    # La confiance moyenne donne une idée de la "certitude" générale du
    # modèle (attention : softmax est naturellement sur-confiant).
    print(f"  confiance moyenne : {confidences.mean():.4f}")

    # Rapport détaillé PAR classe : permet de voir si le modèle est meilleur
    # sur les positifs que sur les négatifs (ou l'inverse).
    print("\nRapport par classe :")
    print(classification_report(
        labels, predictions, target_names=["négatif", "positif"], digits=4
    ))

    # Matrice de confusion : la diagonale = succès, hors-diagonale = erreurs
    matrix = confusion_matrix(labels, predictions)
    print("Matrice de confusion :")
    print(f"{'':>16} | {'prédit négatif':>15} | {'prédit positif':>15}")
    print("-" * 52)
    print(f"{'vrai négatif':>16} | {matrix[0][0]:>15} | {matrix[0][1]:>15}")
    print(f"{'vrai positif':>16} | {matrix[1][0]:>15} | {matrix[1][1]:>15}")

    # ── 5. Analyse qualitative des erreurs ──────────────────────────────────
    show_most_confident_errors(
        texts, labels, predictions, confidences, id2label, args.show_errors
    )


if __name__ == "__main__":
    main()
