"""
train.py — Fine-tuning de DistilCamemBERT sur le dataset Allociné (avis ciné).
===============================================================================

Tâche : classification binaire d'avis de films en français.
    label 0 = négatif | label 1 = positif

DÉCISION ARCHITECT — deux stratégies de transfer learning sont implémentées :

    Stratégie A — FINE-TUNING COMPLET (par défaut)
        Tous les poids du modèle sont ajustés, avec un learning rate très
        faible (2e-5). Le backbone adapte sa représentation du français au
        vocabulaire des critiques de cinéma.
        -> Meilleure performance (~0.94 F1), mais plus de calcul.

    Stratégie B — BACKBONE GELÉ, TÊTE SEULE (--freeze-backbone)
        Les 68M de paramètres de DistilCamemBERT sont gelés (requires_grad
        = False) : seule la tête de classification (~0.6M de paramètres)
        est entraînée. Le backbone sert d'extracteur de features figé.
        -> ~3x plus rapide, quasi possible sur CPU, mais performance
           inférieure (~0.85-0.89 F1) : les features génériques du
           pré-entraînement ne sont pas adaptées à la tâche.

    Lancer les deux et comparer les training_metrics.json : c'est cet
    arbitrage chiffré qui justifie le choix final du fine-tuning complet.

Déroulement du script :
    1. Chargement du dataset Allociné depuis le Hugging Face Hub
    2. Exploration rapide : tailles, distribution des labels, exemples
    3. Sous-échantillonnage + tokenisation
    4. Chargement du modèle + application de la stratégie (gel ou non)
    5. Entraînement avec le Trainer de Hugging Face
    6. Évaluation finale sur le set de TEST + matrice de confusion
    7. Sauvegarde du modèle, du tokenizer et des métriques (JSON)

Exécution (depuis la racine du projet) :
    # Stratégie A — fine-tuning complet (choix final du projet) :
    python scripts/train.py

    # Stratégie B — backbone gelé, pour la comparaison :
    python scripts/train.py --freeze-backbone --learning-rate 1e-3

    # Version rapide pour tester le pipeline sur CPU (~10 min) :
    python scripts/train.py --max-train 2000 --max-val 500 --max-test 500 --epochs 1

Sur Google Colab (GPU gratuit, recommandé) :
    !pip install -q transformers datasets accelerate scikit-learn
    !python scripts/train.py

Le modèle est sauvegardé dans : model/sentiment_model/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Constantes du projet
# ---------------------------------------------------------------------------
# DistilCamemBERT : version distillée de CamemBERT (6 couches au lieu de 12).
# ~2x plus rapide pour ~97 % de la performance — adapté à un déploiement
# sur CPU (Hugging Face Spaces gratuit).
MODEL_NAME = "cmarkea/distilcamembert-base"

# Convention de labels du dataset Allociné. Enregistrée dans la config du
# modèle (id2label/label2id) pour que l'API renvoie directement
# "positif"/"négatif" au lieu de "LABEL_0"/"LABEL_1".
ID2LABEL = {0: "négatif", 1: "positif"}
LABEL2ID = {"négatif": 0, "positif": 1}

# Longueur maximale en tokens. 256 couvre ~95 % des avis Allociné ;
# les avis plus longs sont tronqués (la fin est ignorée).
MAX_LENGTH = 256

# Graine aléatoire fixe : shuffle, initialisation de la tête et ordre des
# batchs reproductibles d'un run à l'autre.
SEED = 42

# Racine du projet (= dossier parent de scripts/) : les chemins fonctionnent
# quel que soit le dossier depuis lequel on lance le script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "model" / "sentiment_model"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# 0. Arguments en ligne de commande
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Hyperparamètres et stratégie ajustables sans modifier le code."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning DistilCamemBERT sur Allociné"
    )
    parser.add_argument("--max-train", type=int, default=8000,
                        help="Nombre d'avis pour l'entraînement (max 160 000)")
    parser.add_argument("--max-val", type=int, default=2000,
                        help="Nombre d'avis pour la validation (max 20 000)")
    parser.add_argument("--max-test", type=int, default=2000,
                        help="Nombre d'avis pour l'évaluation finale (max 20 000)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Nombre d'époques (2 suffisent en fine-tuning complet)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Taille de batch (réduire à 8 si mémoire GPU insuffisante)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate. 2e-5 pour le fine-tuning complet ; "
                             "monter à ~1e-3 avec --freeze-backbone (la tête "
                             "part de zéro, elle peut apprendre plus vite)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Stratégie B : gèle DistilCamemBERT et n'entraîne "
                             "que la tête de classification")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1 & 2. Chargement + exploration du dataset
# ---------------------------------------------------------------------------
def load_allocine():
    """Charge le dataset Allociné (avis de cinéma) et vérifie sa structure.

    Pourquoi Allociné ? Dataset de référence du sentiment en français :
    ~200 000 avis de films réels, labels dérivés des notes utilisateurs
    (donc fiables), classes équilibrées, et 3 splits déjà séparés — aucun
    risque de fuite de données entre entraînement et évaluation :
        train      : 160 000 avis
        validation :  20 000 avis  (suivi pendant l'entraînement)
        test       :  20 000 avis  (évaluation finale uniquement)
    Colonnes : "review" (texte de l'avis) et "label" (0 ou 1).
    """
    print("=" * 70)
    print("ÉTAPE 1/5 — Chargement du dataset Allociné")
    print("=" * 70)
    dataset = load_dataset("allocine")

    # Vérification défensive des colonnes (au cas où le dataset évoluerait)
    expected_columns = {"review", "label"}
    actual_columns = set(dataset["train"].column_names)
    if not expected_columns.issubset(actual_columns):
        raise ValueError(
            f"Colonnes attendues : {expected_columns}, trouvées : {actual_columns}"
        )

    # --- Exploration rapide -------------------------------------------------
    print(f"\nSplits disponibles : {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        labels = split_data["label"]
        n_positive = sum(labels)
        n_total = len(labels)
        print(f"  {split_name:<12} : {n_total:>7} avis "
              f"({n_positive / n_total:.1%} positifs — dataset équilibré)")

    # Un exemple de chaque classe pour visualiser les données
    print("\nExemple d'avis POSITIF (label=1) :")
    positive_example = next(ex for ex in dataset["train"] if ex["label"] == 1)
    print(f"  « {positive_example['review'][:150]}... »")
    print("Exemple d'avis NÉGATIF (label=0) :")
    negative_example = next(ex for ex in dataset["train"] if ex["label"] == 0)
    print(f"  « {negative_example['review'][:150]}... »")

    return dataset


# ---------------------------------------------------------------------------
# 3. Sous-échantillonnage + tokenisation
# ---------------------------------------------------------------------------
def prepare_datasets(dataset, tokenizer, max_train: int, max_val: int, max_test: int):
    """Sous-échantillonne chaque split puis tokenise les textes.

    Pourquoi sous-échantillonner ? Grâce au transfer learning, le modèle
    connaît déjà le français : 8 000 exemples suffisent pour apprendre la
    frontière positif/négatif (~94 % de F1). Les 160 000 avis complets
    n'apporteraient que 1-2 points de plus pour 20x plus de calcul.

    Le shuffle AVANT la sélection garantit un échantillon aléatoire donc
    représentatif (le dataset d'origine est équilibré ~50/50).
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 2/5 — Sous-échantillonnage et tokenisation")
    print("=" * 70)

    train_data = dataset["train"].shuffle(seed=SEED).select(range(max_train))
    val_data = dataset["validation"].shuffle(seed=SEED).select(range(max_val))
    test_data = dataset["test"].shuffle(seed=SEED).select(range(max_test))
    print(f"Train : {len(train_data)} | Validation : {len(val_data)} "
          f"| Test : {len(test_data)}")

    def tokenize_batch(batch):
        """Convertit un lot de textes en identifiants de tokens.

        La tokenisation découpe chaque avis en sous-mots (SentencePiece),
        puis mappe chaque sous-mot vers un entier du vocabulaire :
            "Ce film est génial" -> ["▁Ce", "▁film", "▁est", "▁génial"]
                                 -> [149, 1621, 30, 11197]
        truncation=True coupe à MAX_LENGTH tokens. Le padding n'est PAS
        fait ici : il est ajouté dynamiquement par batch pendant
        l'entraînement (DataCollatorWithPadding), plus efficace que de
        padder tous les avis à 256 tokens dès maintenant.
        """
        return tokenizer(batch["review"], truncation=True, max_length=MAX_LENGTH)

    # batched=True : la fonction reçoit des lots de 1000 exemples au lieu
    # d'un seul -> tokenisation beaucoup plus rapide.
    # remove_columns : le texte brut n'est plus utile après tokenisation.
    train_data = train_data.map(tokenize_batch, batched=True, remove_columns=["review"])
    val_data = val_data.map(tokenize_batch, batched=True, remove_columns=["review"])
    test_data = test_data.map(tokenize_batch, batched=True, remove_columns=["review"])

    # Vérification : à quoi ressemble un exemple tokenisé ?
    sample = train_data[0]
    print(f"\nExemple tokenisé : {len(sample['input_ids'])} tokens")
    print(f"  input_ids (10 premiers) : {sample['input_ids'][:10]}")
    print(f"  décodés : {tokenizer.convert_ids_to_tokens(sample['input_ids'][:10])}")

    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# 4. Application de la stratégie de transfer learning
# ---------------------------------------------------------------------------
def apply_training_strategy(model, freeze_backbone: bool) -> str:
    """Applique la décision architect : backbone gelé ou fine-tuning complet.

    Geler = mettre requires_grad à False : PyTorch ne calcule plus de
    gradients pour ces poids, ils ne bougent plus pendant l'entraînement.
    Seule la tête de classification (model.classifier, initialisée
    aléatoirement) reste entraînable.
    """
    if freeze_backbone:
        # model.base_model = le corps DistilCamemBERT (embeddings + 6 couches
        # d'attention). La tête model.classifier n'en fait pas partie.
        for param in model.base_model.parameters():
            param.requires_grad = False
        strategy = "head-only (backbone gelé)"
    else:
        strategy = "fine-tuning complet"

    # Bilan chiffré : combien de paramètres vont réellement apprendre ?
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStratégie : {strategy}")
    print(f"  Paramètres totaux      : {total_params / 1e6:>6.1f}M")
    print(f"  Paramètres entraînables : {trainable_params / 1e6:>6.1f}M "
          f"({trainable_params / total_params:.1%})")

    return strategy


# ---------------------------------------------------------------------------
# Métriques d'évaluation
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Calcule accuracy, precision, recall et F1.

    Appelée automatiquement par le Trainer à chaque évaluation.
    eval_pred contient :
        logits : scores bruts du modèle, shape (n_exemples, 2)
        labels : vraies classes, shape (n_exemples,)
    La prédiction = la classe avec le logit le plus élevé (argmax).
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"  # binaire : classe positive = 1
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_confusion_matrix(labels, predictions):
    """Affiche la matrice de confusion en mode texte.

                        prédit négatif   prédit positif
        vrai négatif          VN               FP
        vrai positif          FN               VP
    """
    matrix = confusion_matrix(labels, predictions)
    print("\nMatrice de confusion (test) :")
    print(f"{'':>16} | {'prédit négatif':>15} | {'prédit positif':>15}")
    print("-" * 52)
    print(f"{'vrai négatif':>16} | {matrix[0][0]:>15} | {matrix[0][1]:>15}")
    print(f"{'vrai positif':>16} | {matrix[1][0]:>15} | {matrix[1][1]:>15}")


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    use_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    print(f"Matériel utilisé : {device_name}")
    if not use_gpu:
        print("(Pas de GPU détecté — pensez à Google Colab, ou utilisez "
              "--freeze-backbone / --max-train réduit pour aller plus vite.)")

    # Avertissement pédagogique : tête seule + LR de fine-tuning = sous-optimal
    if args.freeze_backbone and args.learning_rate < 1e-4:
        print(f"\nAttention : --freeze-backbone avec un LR de {args.learning_rate} "
              "est très lent à converger.\nLa tête part de zéro, elle supporte "
              "un LR bien plus grand : recommandé --learning-rate 1e-3")

    # --- 1 & 2. Dataset ------------------------------------------------------
    dataset = load_allocine()

    # --- 3. Tokenizer + préparation des données ------------------------------
    # Le tokenizer DOIT être celui du modèle pré-entraîné : son vocabulaire
    # de sous-mots correspond exactement aux embeddings appris par le modèle.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data, val_data, test_data = prepare_datasets(
        dataset, tokenizer, args.max_train, args.max_val, args.max_test
    )

    # --- 4. Modèle : transfer learning ---------------------------------------
    print("\n" + "=" * 70)
    print("ÉTAPE 3/5 — Chargement du modèle + stratégie d'entraînement")
    print("=" * 70)
    # AutoModelForSequenceClassification = corps de DistilCamemBERT
    # (68M de paramètres qui "savent le français", pré-entraînés sur 138 Go
    # de texte) + une NOUVELLE couche de classification à 2 sorties,
    # initialisée aléatoirement. Le warning "weights newly initialized"
    # affiché par transformers est donc NORMAL : c'est la tête neuve.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Décision architect : gel du backbone ou fine-tuning complet
    strategy = apply_training_strategy(model, args.freeze_backbone)

    # --- 5. Configuration et lancement de l'entraînement ---------------------
    print("\n" + "=" * 70)
    print("ÉTAPE 4/5 — Entraînement")
    print("=" * 70)
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),       # Checkpoints intermédiaires
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # Pas de gradients en éval
        # En fine-tuning complet, le LR est volontairement très faible
        # (100x plus petit qu'un entraînement from scratch) : on AJUSTE des
        # poids déjà bons, on ne veut pas détruire la connaissance du
        # français acquise au pré-entraînement (catastrophic forgetting).
        learning_rate=args.learning_rate,
        weight_decay=0.01,                     # Régularisation L2 contre l'overfitting
        warmup_ratio=0.1,                      # LR monte progressivement sur les premiers 10 %
        eval_strategy="epoch",                 # Évaluation sur la validation à chaque époque
        save_strategy="epoch",
        load_best_model_at_end=True,           # On repart du MEILLEUR checkpoint...
        metric_for_best_model="f1",            # ...selon le F1 de validation
        fp16=use_gpu,                          # Précision mixte : ~2x plus rapide sur GPU
        logging_steps=50,                      # Affiche la loss tous les 50 pas
        report_to="none",                      # Pas de tracking externe (wandb...)
        seed=SEED,
    )

    # DataCollator : assemble les exemples en batchs et padde chaque batch
    # à la longueur de son avis le plus long (padding dynamique).
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    train_runtime = train_result.metrics["train_runtime"]
    print(f"\nEntraînement terminé en {train_runtime:.0f} s")

    # --- 6. Évaluation finale sur le set de TEST -----------------------------
    # Le test set n'a JAMAIS été vu pendant l'entraînement ni servi à choisir
    # le meilleur checkpoint : c'est la mesure honnête de la généralisation.
    print("\n" + "=" * 70)
    print("ÉTAPE 5/5 — Évaluation finale sur le set de test")
    print("=" * 70)
    test_output = trainer.predict(test_data)
    test_metrics = test_output.metrics
    test_predictions = np.argmax(test_output.predictions, axis=-1)

    print("\nMétriques sur le set de TEST :")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"  {name:<10} : {test_metrics[f'test_{name}']:.4f}")
    print_confusion_matrix(test_output.label_ids, test_predictions)

    # --- 7. Sauvegarde --------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))          # Poids + config (id2label inclus)
    tokenizer.save_pretrained(str(OUTPUT_DIR))   # Vocabulaire du tokenizer

    # Métriques + hyperparamètres dans un JSON : traçabilité du run, et
    # c'est ce fichier qu'on compare entre les deux stratégies pour
    # justifier la décision architect.
    metrics_summary = {
        "model": MODEL_NAME,
        "strategy": strategy,
        "train_size": args.max_train,
        "val_size": args.max_val,
        "test_size": args.max_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_runtime_seconds": round(train_runtime, 1),
        "test_accuracy": round(test_metrics["test_accuracy"], 4),
        "test_precision": round(test_metrics["test_precision"], 4),
        "test_recall": round(test_metrics["test_recall"], 4),
        "test_f1": round(test_metrics["test_f1"], 4),
    }
    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    print(f"\nModèle sauvegardé dans   : {OUTPUT_DIR}")
    print(f"Métriques sauvegardées   : {metrics_path}")
    print("\nÉtape suivante : tester avec  python scripts/predict.py \"Votre avis ici\"")


if __name__ == "__main__":
    main()
