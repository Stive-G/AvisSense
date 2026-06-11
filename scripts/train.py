"""
train.py — Fine-tuning de DistilCamemBERT sur le dataset Allociné (avis ciné).
===============================================================================

CE QUE FAIT CE SCRIPT, EN UNE PHRASE :
    Il prend un modèle qui "sait déjà le français" (DistilCamemBERT) et lui
    apprend une tâche précise : décider si un avis de film est positif ou
    négatif.

─────────────────────────────────────────────────────────────────────────────
CONCEPT CLÉ N°1 — LE TRANSFER LEARNING
─────────────────────────────────────────────────────────────────────────────
Entraîner un modèle de langue "from scratch" (à partir de zéro) demanderait
des dizaines de Go de texte et des semaines de GPU. À la place :

    1. Quelqu'un (l'équipe CamemBERT) a PRÉ-ENTRAÎNÉ un gros modèle sur
       138 Go de texte français, avec une tâche générique : deviner des mots
       masqués dans des phrases ("Le chat mange la <MASK>"). En faisant ça
       des milliards de fois, le modèle a appris la grammaire, le vocabulaire
       et le sens des mots français.
    2. Nous, on RÉCUPÈRE ce modèle déjà intelligent, on lui ajoute une petite
       "tête de classification" (une couche de neurones à 2 sorties :
       négatif / positif), et on ajuste le tout sur NOS données (des avis de
       films étiquetés). C'est le FINE-TUNING.

    Résultat : en ~20 minutes et avec seulement 8 000 exemples, on obtient
    ~94 % de réussite. From scratch, il faudrait des millions d'exemples.

─────────────────────────────────────────────────────────────────────────────
CONCEPT CLÉ N°2 — LA DÉCISION ARCHITECT (le cœur du sujet)
─────────────────────────────────────────────────────────────────────────────
Deux façons de faire du transfer learning, TOUTES DEUX implémentées ici :

    Stratégie A — FINE-TUNING COMPLET (par défaut)
        On ajuste TOUS les poids du modèle (les 68 millions), avec un
        learning rate très faible pour ne pas casser ce qu'il sait déjà.
        -> Meilleure performance (~0.94 F1) : le modèle adapte même sa
           compréhension du français au vocabulaire des critiques de ciné.

    Stratégie B — BACKBONE GELÉ (option --freeze-backbone)
        On "gèle" le corps du modèle (ses poids ne bougent plus du tout) et
        on n'entraîne QUE la petite tête de classification (~600 000
        paramètres, soit 1 % du total). Le modèle gelé sert juste
        d'extracteur de features (de représentations numériques du texte).
        -> ~3x plus rapide à entraîner, mais moins bon (~0.85-0.89 F1) car
           les représentations génériques ne sont pas adaptées à la tâche.

    Pour trancher : lancer les deux, comparer les training_metrics.json.

─────────────────────────────────────────────────────────────────────────────
COMMENT LANCER (depuis la racine du projet) :
    # Stratégie A — fine-tuning complet (choix final du projet) :
    python scripts/train.py

    # Stratégie B — backbone gelé, pour la comparaison :
    python scripts/train.py --freeze-backbone --learning-rate 1e-3

    # Version rapide pour vérifier que tout marche, sur CPU (~10 min) :
    python scripts/train.py --max-train 2000 --max-val 500 --max-test 500 --epochs 1

Sur Google Colab (GPU gratuit, recommandé) :
    !pip install -q transformers datasets accelerate scikit-learn
    !python scripts/train.py

SORTIE : le modèle entraîné est sauvegardé dans model/sentiment_model/
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import argparse        # Lire les options passées en ligne de commande (--epochs, etc.)
import json            # Sauvegarder les métriques dans un fichier .json lisible
from pathlib import Path  # Manipuler les chemins de fichiers proprement (Windows/Linux)

import numpy as np     # Calculs sur tableaux (argmax sur les prédictions)
import torch           # PyTorch : le moteur de calcul du deep learning
from datasets import load_dataset  # Télécharge les datasets du Hugging Face Hub
from sklearn.metrics import (      # Les métriques d'évaluation classiques
    accuracy_score,                #   % de bonnes réponses
    confusion_matrix,              #   tableau vrais/faux positifs/négatifs
    precision_recall_fscore_support,  # precision, recall, F1 d'un coup
)
from transformers import (
    AutoModelForSequenceClassification,  # Modèle pré-entraîné + tête de classification
    AutoTokenizer,                       # Convertit le texte en nombres
    DataCollatorWithPadding,             # Assemble les exemples en batchs égalisés
    Trainer,                             # Boucle d'entraînement toute faite de HF
    TrainingArguments,                   # Tous les hyperparamètres de l'entraînement
)

# ─── CONSTANTES DU PROJET ───────────────────────────────────────────────────

# Le modèle pré-entraîné qu'on va fine-tuner.
# "Distil" = version DISTILLÉE : un petit modèle (6 couches) a été entraîné à
# imiter le grand CamemBERT (12 couches). Il garde ~97 % de la performance
# pour 2x moins de calcul. Crucial pour servir le modèle sur un CPU gratuit.
MODEL_NAME = "cmarkea/distilcamembert-base"

# La correspondance numéro de classe <-> nom lisible.
# Le dataset Allociné utilise : 0 = négatif, 1 = positif.
# On enregistre ce mapping DANS le modèle (voir plus bas) pour que l'API
# renvoie directement "positif"/"négatif" au lieu de "LABEL_0"/"LABEL_1".
ID2LABEL = {0: "négatif", 1: "positif"}
LABEL2ID = {"négatif": 0, "positif": 1}

# Longueur maximale d'un avis, EN TOKENS (pas en caractères !).
# Un token ≈ un sous-mot (voir la fonction tokenize_batch plus bas).
# 256 tokens couvrent ~95 % des avis Allociné. Les avis plus longs sont
# TRONQUÉS : on ne garde que les 256 premiers tokens, la fin est ignorée.
# Pourquoi pas plus ? Le coût de calcul d'un transformer croît avec le CARRÉ
# de la longueur (mécanisme d'attention) : 512 tokens = 4x plus cher que 256.
MAX_LENGTH = 256

# Graine aléatoire fixe pour la REPRODUCTIBILITÉ : le mélange du dataset,
# l'initialisation de la tête et l'ordre des batchs seront identiques à
# chaque exécution -> on peut comparer deux runs de manière fiable.
SEED = 42

# Chemins ABSOLUS calculés à partir de l'emplacement de ce fichier :
# __file__ = ce script -> .parent = scripts/ -> .parent = la racine du projet.
# Avantage : le script marche quel que soit le dossier d'où on le lance.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "model" / "sentiment_model"   # Modèle final
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"            # Sauvegardes intermédiaires


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 0 — Lire les options de la ligne de commande
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Définit les options ajustables sans toucher au code.

    Exemple : `python scripts/train.py --epochs 3 --batch-size 8`
    argparse lit ces options et les rend disponibles dans `args.epochs`, etc.
    """
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
                        help="Une ÉPOQUE = un passage complet sur tout le "
                             "dataset d'entraînement. 2 suffisent en fine-tuning : "
                             "au-delà, le modèle commence à mémoriser (overfitting)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Nombre d'avis traités EN MÊME TEMPS à chaque pas. "
                             "Plus grand = plus rapide mais plus de mémoire GPU. "
                             "Réduire à 8 si erreur 'out of memory'")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Taille des pas d'apprentissage. 2e-5 (=0.00002) pour "
                             "le fine-tuning complet ; monter à ~1e-3 avec "
                             "--freeze-backbone (la tête part de zéro, elle "
                             "peut apprendre plus vite sans rien casser)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Stratégie B : gèle DistilCamemBERT et n'entraîne "
                             "que la tête de classification")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPES 1 & 2 — Charger et explorer le dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_allocine():
    """Charge le dataset Allociné (avis de cinéma) et vérifie sa structure.

    POURQUOI ALLOCINÉ ?
    - Dataset de référence du sentiment en français : ~200 000 avis de films
      réels écrits par des internautes.
    - Les labels viennent des NOTES laissées par les utilisateurs (étoiles),
      pas d'une annotation manuelle -> fiables et peu bruités.
    - Classes équilibrées ~50/50 -> pas besoin de techniques de rééquilibrage.
    - 3 splits DÉJÀ séparés -> aucun risque de "fuite de données" (un même
      avis qui serait à la fois dans le train et dans le test fausserait
      l'évaluation) :
        train      : 160 000 avis  (pour apprendre)
        validation :  20 000 avis  (pour surveiller l'apprentissage)
        test       :  20 000 avis  (pour la note finale, jamais vu avant)

    Chaque exemple a 2 colonnes : "review" (le texte) et "label" (0 ou 1).
    """
    print("=" * 70)
    print("ÉTAPE 1/5 — Chargement du dataset Allociné")
    print("=" * 70)
    # load_dataset télécharge le dataset depuis le Hub la 1re fois (~64 Mo),
    # puis le lit depuis le cache local (~/.cache/huggingface) ensuite.
    dataset = load_dataset("allocine")

    # Vérification défensive : si le dataset change un jour de format, on
    # préfère une erreur claire ici plutôt qu'un plantage obscur plus loin.
    expected_columns = {"review", "label"}
    actual_columns = set(dataset["train"].column_names)
    if not expected_columns.issubset(actual_columns):
        raise ValueError(
            f"Colonnes attendues : {expected_columns}, trouvées : {actual_columns}"
        )

    # ── Exploration rapide : connaître ses données AVANT d'entraîner ───────
    print(f"\nSplits disponibles : {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        labels = split_data["label"]
        n_positive = sum(labels)   # Labels 0/1 : la somme = le nb de positifs
        n_total = len(labels)
        print(f"  {split_name:<12} : {n_total:>7} avis "
              f"({n_positive / n_total:.1%} positifs — dataset équilibré)")

    # Afficher un exemple de chaque classe : toujours REGARDER ses données.
    print("\nExemple d'avis POSITIF (label=1) :")
    positive_example = next(ex for ex in dataset["train"] if ex["label"] == 1)
    print(f"  « {positive_example['review'][:150]}... »")
    print("Exemple d'avis NÉGATIF (label=0) :")
    negative_example = next(ex for ex in dataset["train"] if ex["label"] == 0)
    print(f"  « {negative_example['review'][:150]}... »")

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — Sous-échantillonner et tokeniser
# ─────────────────────────────────────────────────────────────────────────────
def prepare_datasets(dataset, tokenizer, max_train: int, max_val: int, max_test: int):
    """Réduit la taille des splits puis convertit les textes en nombres.

    POURQUOI SOUS-ÉCHANTILLONNER (8 000 avis au lieu de 160 000) ?
    Grâce au transfer learning, le modèle connaît déjà le français : il n'a
    besoin que d'apprendre la frontière positif/négatif. La performance
    sature vite : ~94 % de F1 avec 8k exemples. Les 160k complets
    n'apporteraient que 1-2 points de plus... pour 20x plus de calcul.
    C'est un arbitrage coût/bénéfice assumé (et réglable via --max-train).
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 2/5 — Sous-échantillonnage et tokenisation")
    print("=" * 70)

    # .shuffle(seed=...) mélange les avis AVANT d'en prendre les N premiers :
    # on obtient un échantillon ALÉATOIRE donc représentatif. Sans shuffle,
    # on prendrait les N premiers avis du fichier, qui pourraient être triés
    # (par film, par date...) et donc biaisés.
    train_data = dataset["train"].shuffle(seed=SEED).select(range(max_train))
    val_data = dataset["validation"].shuffle(seed=SEED).select(range(max_val))
    test_data = dataset["test"].shuffle(seed=SEED).select(range(max_test))
    print(f"Train : {len(train_data)} | Validation : {len(val_data)} "
          f"| Test : {len(test_data)}")

    def tokenize_batch(batch):
        """Convertit un lot de textes en identifiants de tokens.

        ─── CONCEPT : LA TOKENISATION ───────────────────────────────────────
        Un réseau de neurones ne comprend que des NOMBRES. La tokenisation
        fait la conversion texte -> nombres en 2 temps :

        1. DÉCOUPAGE en sous-mots (algorithme SentencePiece) :
             "Ce film est génial" -> ["▁Ce", "▁film", "▁est", "▁génial"]
           Pourquoi des SOUS-mots et pas des mots entiers ? Pour gérer les
           mots inconnus : "inregardable" n'est pas dans le vocabulaire,
           mais "in" + "regard" + "able" oui. Aucun mot n'est jamais
           vraiment "inconnu". (Le ▁ marque un début de mot.)

        2. CONVERSION de chaque sous-mot en son numéro dans le vocabulaire
           (32 000 entrées) :
             ["▁Ce", "▁film", "▁est", "▁génial"] -> [149, 1621, 30, 11197]

        Le tokenizer ajoute aussi 2 tokens spéciaux : <s> au début (c'est
        SA représentation finale que la tête de classification utilisera
        pour décider) et </s> à la fin.

        truncation=True : coupe à MAX_LENGTH tokens si l'avis est trop long.
        Le PADDING (compléter les avis courts pour égaliser les longueurs)
        n'est PAS fait ici mais plus tard, batch par batch — voir
        DataCollatorWithPadding plus bas.
        ─────────────────────────────────────────────────────────────────────
        """
        return tokenizer(batch["review"], truncation=True, max_length=MAX_LENGTH)

    # .map applique la fonction à tout le dataset.
    # batched=True : la fonction reçoit des lots de 1000 exemples au lieu
    #   d'un seul -> tokenisation beaucoup plus rapide.
    # remove_columns : une fois tokenisé, le texte brut ne sert plus à rien,
    #   on le supprime pour alléger la mémoire.
    train_data = train_data.map(tokenize_batch, batched=True, remove_columns=["review"])
    val_data = val_data.map(tokenize_batch, batched=True, remove_columns=["review"])
    test_data = test_data.map(tokenize_batch, batched=True, remove_columns=["review"])

    # Vérification visuelle : à quoi ressemble un exemple après tokenisation ?
    sample = train_data[0]
    print(f"\nExemple tokenisé : {len(sample['input_ids'])} tokens")
    print(f"  input_ids (10 premiers) : {sample['input_ids'][:10]}")
    print(f"  décodés : {tokenizer.convert_ids_to_tokens(sample['input_ids'][:10])}")

    return train_data, val_data, test_data


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — Appliquer la stratégie de transfer learning (décision architect)
# ─────────────────────────────────────────────────────────────────────────────
def apply_training_strategy(model, freeze_backbone: bool) -> str:
    """Gèle (ou non) le backbone selon la stratégie choisie.

    COMMENT ON "GÈLE" UN MODÈLE ?
    Chaque poids (paramètre) du modèle a un attribut `requires_grad` :
    - True  -> PyTorch calcule son gradient et l'optimiseur le modifie
               à chaque pas d'apprentissage (le poids "apprend").
    - False -> le poids est ignoré par l'entraînement : il reste figé.

    Ici on met requires_grad=False sur tout le CORPS du modèle
    (model.base_model = embeddings + les 6 couches d'attention).
    La tête de classification (model.classifier), elle, n'en fait pas
    partie : elle reste entraînable. Comme elle vient d'être initialisée
    aléatoirement, c'est de toute façon elle qui doit apprendre.
    """
    if freeze_backbone:
        for param in model.base_model.parameters():
            param.requires_grad = False     # Ce poids n'apprendra plus rien
        strategy = "head-only (backbone gelé)"
    else:
        strategy = "fine-tuning complet"    # Tous les poids restent entraînables

    # Bilan chiffré : combien de paramètres vont réellement apprendre ?
    # p.numel() = nombre d'éléments d'un tenseur de poids.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStratégie : {strategy}")
    print(f"  Paramètres totaux       : {total_params / 1e6:>6.1f}M")
    print(f"  Paramètres entraînables : {trainable_params / 1e6:>6.1f}M "
          f"({trainable_params / total_params:.1%})")

    return strategy


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES — comment on mesure la qualité du modèle
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Calcule les 4 métriques classiques de classification.

    Cette fonction est appelée AUTOMATIQUEMENT par le Trainer à chaque
    évaluation. Elle reçoit :
        logits : les scores bruts du modèle, tableau (n_exemples, 2)
                 -> pour chaque avis : [score_négatif, score_positif]
        labels : les vraies réponses, tableau (n_exemples,)

    ─── CONCEPT : LES 4 MÉTRIQUES ──────────────────────────────────────────
    En notant : VP = vrais positifs (prédit positif, c'était positif)
                FP = faux positifs (prédit positif, c'était négatif)
                FN = faux négatifs (prédit négatif, c'était positif)

    accuracy  = % de bonnes réponses au total.
                Suffisante ici car les classes sont équilibrées 50/50.
    precision = VP / (VP + FP)
                "Quand le modèle dit positif, a-t-il raison ?"
    recall    = VP / (VP + FN)
                "Parmi les vrais positifs, combien en a-t-il trouvés ?"
    f1        = moyenne harmonique de precision et recall.
                Résume les deux en un seul chiffre ; c'est notre métrique
                de référence pour choisir le meilleur checkpoint.
    ─────────────────────────────────────────────────────────────────────────
    """
    logits, labels = eval_pred
    # argmax : pour chaque avis, prend l'indice du plus grand score
    # -> [0.2, 1.7] devient 1 (= positif). C'est la prédiction du modèle.
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"  # binaire : la classe "1" est la référence
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_confusion_matrix(labels, predictions):
    """Affiche la matrice de confusion : OÙ le modèle se trompe-t-il ?

                          prédit négatif   prédit positif
        vrai négatif     [ vrais négatifs ][ faux positifs ]
        vrai positif     [ faux négatifs  ][ vrais positifs]

    La diagonale = les bonnes réponses. Hors diagonale = les erreurs.
    Si une case d'erreur est beaucoup plus grosse que l'autre, le modèle a
    un biais (ex : il prédit trop facilement "positif").
    """
    matrix = confusion_matrix(labels, predictions)
    print("\nMatrice de confusion (test) :")
    print(f"{'':>16} | {'prédit négatif':>15} | {'prédit positif':>15}")
    print("-" * 52)
    print(f"{'vrai négatif':>16} | {matrix[0][0]:>15} | {matrix[0][1]:>15}")
    print(f"{'vrai positif':>16} | {matrix[1][0]:>15} | {matrix[1][1]:>15}")


# ─────────────────────────────────────────────────────────────────────────────
# PROGRAMME PRINCIPAL — enchaîne les 5 étapes
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Détecte si un GPU NVIDIA est disponible. Sur GPU, l'entraînement est
    # ~30x plus rapide que sur CPU (calcul matriciel massivement parallèle).
    use_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    print(f"Matériel utilisé : {device_name}")
    if not use_gpu:
        print("(Pas de GPU détecté — pensez à Google Colab, ou utilisez "
              "--freeze-backbone / --max-train réduit pour aller plus vite.)")

    # Garde-fou : avec le backbone gelé, la tête part de zéro et peut
    # apprendre vite -> un LR de fine-tuning (2e-5) serait inutilement lent.
    if args.freeze_backbone and args.learning_rate < 1e-4:
        print(f"\nAttention : --freeze-backbone avec un LR de {args.learning_rate} "
              "est très lent à converger.\nLa tête part de zéro, elle supporte "
              "un LR bien plus grand : recommandé --learning-rate 1e-3")

    # ── ÉTAPES 1 & 2 : le dataset ──────────────────────────────────────────
    dataset = load_allocine()

    # ── ÉTAPE 3 : tokenizer + données ──────────────────────────────────────
    # Le tokenizer DOIT être celui du modèle pré-entraîné : chaque numéro de
    # token correspond à un embedding (vecteur) précis appris par le modèle.
    # Utiliser un autre tokenizer = parler une autre langue au modèle.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data, val_data, test_data = prepare_datasets(
        dataset, tokenizer, args.max_train, args.max_val, args.max_test
    )

    # ── ÉTAPE 4 : le modèle ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3/5 — Chargement du modèle + stratégie d'entraînement")
    print("=" * 70)
    # AutoModelForSequenceClassification assemble :
    #   [corps DistilCamemBERT pré-entraîné : 68M de poids qui "savent le français"]
    #   + [tête de classification : 1 couche linéaire -> 2 sorties, init ALÉATOIRE]
    #
    # transformers affichera un warning "Some weights were newly initialized" :
    # c'est NORMAL et attendu — c'est justement la tête neuve qu'on va entraîner.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,         # Classification binaire -> 2 neurones de sortie
        id2label=ID2LABEL,    # Enregistré dans le modèle : l'API rendra "positif"
        label2id=LABEL2ID,    # et non "LABEL_1"
    )

    # Décision architect : gel du backbone ou fine-tuning complet
    strategy = apply_training_strategy(model, args.freeze_backbone)

    # ── ÉTAPE 5 : configuration de l'entraînement ──────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 4/5 — Entraînement")
    print("=" * 70)
    #
    # ─── CONCEPT : COMMENT UN MODÈLE "APPREND" ──────────────────────────────
    # À chaque pas : (1) le modèle prédit sur un batch d'avis, (2) on mesure
    # son erreur (la "loss" : ici la cross-entropy entre ses probabilités et
    # les vraies réponses), (3) on calcule dans quelle direction modifier
    # chaque poids pour réduire l'erreur (les gradients, par rétropropagation),
    # (4) l'optimiseur (AdamW) déplace chaque poids d'un petit pas dans cette
    # direction. La taille de ce pas, c'est le LEARNING RATE.
    # ─────────────────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        # Où stocker les checkpoints (sauvegardes en cours d'entraînement)
        output_dir=str(CHECKPOINTS_DIR),

        # Nombre de passages complets sur les données d'entraînement
        num_train_epochs=args.epochs,

        # Taille des batchs. En évaluation on peut doubler : pas de gradients
        # à stocker, donc 2x moins de mémoire consommée.
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,

        # LEARNING RATE volontairement minuscule en fine-tuning (2e-5, soit
        # ~100x plus petit qu'un entraînement from scratch) : les poids sont
        # déjà bons, on les AJUSTE délicatement. Un LR trop grand détruirait
        # la connaissance du français du pré-entraînement (phénomène appelé
        # "catastrophic forgetting").
        learning_rate=args.learning_rate,

        # WEIGHT DECAY : régularisation L2. Tire légèrement tous les poids
        # vers zéro à chaque pas -> décourage les poids extrêmes -> le modèle
        # généralise mieux au lieu de mémoriser (anti-overfitting).
        weight_decay=0.01,

        # WARMUP : pendant les premiers 10 % des pas, le LR monte
        # progressivement de 0 à sa valeur cible. Évite que les premiers
        # gradients (très bruités car la tête est aléatoire) ne fassent
        # faire n'importe quoi au modèle.
        warmup_ratio=0.1,

        # Évaluer sur la VALIDATION à la fin de chaque époque, et sauvegarder
        # un checkpoint au même moment.
        eval_strategy="epoch",
        save_strategy="epoch",

        # À la fin, recharger automatiquement le MEILLEUR checkpoint (et pas
        # forcément le dernier !) selon le F1 de validation. Si l'époque 2
        # overfitte, on repart de l'époque 1.
        load_best_model_at_end=True,
        metric_for_best_model="f1",

        # FP16 (précision mixte) : calculs en nombres 16 bits au lieu de 32
        # quand c'est possible -> ~2x plus rapide et 2x moins de mémoire sur
        # GPU, sans perte de qualité notable. Inutile/inactif sur CPU.
        fp16=use_gpu,

        # Afficher la loss toutes les 50 itérations pour suivre la convergence
        # (elle doit DESCENDRE ; si elle stagne ou remonte, problème).
        logging_steps=50,

        # Pas d'envoi de logs vers des services externes (wandb, tensorboard...)
        report_to="none",

        seed=SEED,
    )

    # ─── CONCEPT : LE PADDING DYNAMIQUE ─────────────────────────────────────
    # Les avis d'un même batch n'ont pas la même longueur, or un batch doit
    # être un tableau rectangulaire. Le DataCollator complète ("padde") les
    # avis courts avec un token spécial <pad> jusqu'à la longueur du PLUS
    # LONG avis DU BATCH (et pas 256 systématiquement) -> beaucoup moins de
    # calcul gaspillé sur du remplissage.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Le Trainer encapsule toute la boucle d'entraînement : itération sur les
    # batchs, calcul de la loss, rétropropagation, optimiseur, évaluations,
    # checkpoints... Sans lui, il faudrait écrire ~100 lignes de PyTorch.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # C'est ICI que tout se passe (la seule ligne qui calcule pendant ~20 min)
    train_result = trainer.train()
    train_runtime = train_result.metrics["train_runtime"]
    print(f"\nEntraînement terminé en {train_runtime:.0f} s")

    # ── ÉTAPE 6 : évaluation finale sur le TEST ────────────────────────────
    # POURQUOI UN 3e JEU DE DONNÉES ? La validation a servi à CHOISIR le
    # meilleur checkpoint -> le modèle l'a indirectement "vue". Le test, lui,
    # n'a influencé AUCUNE décision : c'est la mesure honnête de ce que le
    # modèle vaudra sur des avis réellement nouveaux (la généralisation).
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

    # ── ÉTAPE 7 : sauvegarde ───────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # save_model écrit les poids (model.safetensors, ~270 Mo) + la config
    # (config.json, qui contient notre mapping id2label).
    trainer.save_model(str(OUTPUT_DIR))
    # Le tokenizer aussi : en inférence il faut découper le texte EXACTEMENT
    # de la même façon qu'à l'entraînement.
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Trace écrite du run : hyperparamètres + résultats dans un JSON.
    # C'est CE fichier qu'on compare entre les deux stratégies (champ
    # "strategy") pour justifier la décision architect avec des chiffres.
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
        # ensure_ascii=False : garde les accents lisibles dans le JSON
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    print(f"\nModèle sauvegardé dans   : {OUTPUT_DIR}")
    print(f"Métriques sauvegardées   : {metrics_path}")
    print("\nÉtape suivante : tester avec  python scripts/predict.py \"Votre avis ici\"")


# Point d'entrée standard Python : ce bloc ne s'exécute que si on lance le
# fichier directement (python scripts/train.py), pas si on l'importe.
if __name__ == "__main__":
    main()
