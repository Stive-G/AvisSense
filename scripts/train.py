"""Fine-tuning de DistilCamemBERT sur le dataset Allociné."""

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))

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

MODEL_NAME = "cmarkea/distilcamembert-base"
ID2LABEL = {0: "négatif", 1: "positif"}
LABEL2ID = {"négatif": 0, "positif": 1}
MAX_LENGTH = 256
SEED = 42
OUTPUT_DIR = PROJECT_ROOT / "model" / "sentiment_model"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning DistilCamemBERT sur Allociné")
    parser.add_argument("--max-train", type=int, default=8000, help="Nombre d'avis pour l'entraînement")
    parser.add_argument("--max-val", type=int, default=2000, help="Nombre d'avis pour la validation")
    parser.add_argument("--max-test", type=int, default=2000, help="Nombre d'avis pour l'évaluation finale")
    parser.add_argument("--epochs", type=int, default=2, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille des batchs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Gèle le backbone et n'entraîne que la tête de classification",
    )
    return parser.parse_args()


def load_allocine():
    """Charge Allociné et vérifie sa structure."""
    print("=" * 70)
    print("ÉTAPE 1/5 — Chargement du dataset Allociné")
    print("=" * 70)
    dataset = load_dataset("allocine")

    expected_columns = {"review", "label"}
    actual_columns = set(dataset["train"].column_names)
    if not expected_columns.issubset(actual_columns):
        raise ValueError(f"Colonnes attendues : {expected_columns}, trouvées : {actual_columns}")

    print(f"\nSplits disponibles : {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        labels = split_data["label"]
        n_positive = sum(labels)
        n_total = len(labels)
        print(
            f"  {split_name:<12} : {n_total:>7} avis "
            f"({n_positive / n_total:.1%} positifs)"
        )

    positive_example = next(ex for ex in dataset["train"] if ex["label"] == 1)
    negative_example = next(ex for ex in dataset["train"] if ex["label"] == 0)
    print("\nExemple d'avis POSITIF (label=1) :")
    print(f"  « {positive_example['review'][:150]}... »")
    print("Exemple d'avis NÉGATIF (label=0) :")
    print(f"  « {negative_example['review'][:150]}... »")

    return dataset


def prepare_datasets(dataset, tokenizer, max_train: int, max_val: int, max_test: int):
    """Sous-échantillonne et tokenise les splits."""
    print("\n" + "=" * 70)
    print("ÉTAPE 2/5 — Sous-échantillonnage et tokenisation")
    print("=" * 70)

    train_data = dataset["train"].shuffle(seed=SEED).select(range(max_train))
    val_data = dataset["validation"].shuffle(seed=SEED).select(range(max_val))
    test_data = dataset["test"].shuffle(seed=SEED).select(range(max_test))
    print(f"Train : {len(train_data)} | Validation : {len(val_data)} | Test : {len(test_data)}")

    def tokenize_batch(batch):
        return tokenizer(batch["review"], truncation=True, max_length=MAX_LENGTH)

    train_data = train_data.map(tokenize_batch, batched=True, remove_columns=["review"])
    val_data = val_data.map(tokenize_batch, batched=True, remove_columns=["review"])
    test_data = test_data.map(tokenize_batch, batched=True, remove_columns=["review"])

    sample = train_data[0]
    print(f"\nExemple tokenisé : {len(sample['input_ids'])} tokens")
    print(f"  input_ids (10 premiers) : {sample['input_ids'][:10]}")
    print(f"  décodés : {tokenizer.convert_ids_to_tokens(sample['input_ids'][:10])}")

    return train_data, val_data, test_data


def apply_training_strategy(model, freeze_backbone: bool) -> str:
    """Applique la stratégie de fine-tuning choisie."""
    if freeze_backbone:
        for param in model.base_model.parameters():
            param.requires_grad = False
        strategy = "head-only (backbone gelé)"
    else:
        strategy = "fine-tuning complet"

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"\nStratégie : {strategy}")
    print(f"  Paramètres totaux       : {total_params / 1e6:>6.1f}M")
    print(
        f"  Paramètres entraînables : {trainable_params / 1e6:>6.1f}M "
        f"({trainable_params / total_params:.1%})"
    )
    return strategy


def compute_metrics(eval_pred):
    """Calcule accuracy, precision, recall et F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_confusion_matrix(labels, predictions):
    """Affiche la matrice de confusion du test."""
    matrix = confusion_matrix(labels, predictions)
    print("\nMatrice de confusion (test) :")
    print(f"{'':>16} | {'prédit négatif':>15} | {'prédit positif':>15}")
    print("-" * 52)
    print(f"{'vrai négatif':>16} | {matrix[0][0]:>15} | {matrix[0][1]:>15}")
    print(f"{'vrai positif':>16} | {matrix[1][0]:>15} | {matrix[1][1]:>15}")


def main():
    args = parse_args()

    use_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    print(f"Matériel utilisé : {device_name}")
    if not use_gpu:
        print(
            "(Pas de GPU détecté — pensez à Google Colab, ou utilisez "
            "--freeze-backbone / --max-train réduit pour aller plus vite.)"
        )

    if args.freeze_backbone and args.learning_rate < 1e-4:
        print(
            f"\nAttention : --freeze-backbone avec un learning rate de {args.learning_rate} "
            "sera lent à converger.\nRecommandation : --learning-rate 1e-3"
        )

    dataset = load_allocine()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data, val_data, test_data = prepare_datasets(
        dataset, tokenizer, args.max_train, args.max_val, args.max_test
    )

    print("\n" + "=" * 70)
    print("ÉTAPE 3/5 — Chargement du modèle et stratégie d'entraînement")
    print("=" * 70)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    strategy = apply_training_strategy(model, args.freeze_backbone)

    print("\n" + "=" * 70)
    print("ÉTAPE 4/5 — Entraînement")
    print("=" * 70)
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_gpu,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

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
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_summary, file, indent=2, ensure_ascii=False)

    print(f"\nModèle sauvegardé dans   : {OUTPUT_DIR}")
    print(f"Métriques sauvegardées   : {metrics_path}")
    print('\nÉtape suivante : tester avec  python scripts/predict.py "Votre avis ici"')


if __name__ == "__main__":
    main()
