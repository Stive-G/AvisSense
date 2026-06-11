"""
predict.py — Prédiction du sentiment d'un avis avec le modèle fine-tuné.
=========================================================================

CE QUE FAIT CE SCRIPT :
    Il recharge le modèle entraîné par train.py et prédit le sentiment d'un
    avis. L'inférence est faite "à la main" (sans la pipeline toute faite de
    Hugging Face) pour rendre VISIBLES les 4 étapes d'une prédiction :

    ┌──────────────────────────────────────────────────────────────────┐
    │ 1. TOKENISATION : le texte devient une suite de nombres           │
    │      "Super film"  ->  [5, 1234, 1621, 6]                          │
    │ 2. FORWARD PASS : le modèle produit 2 LOGITS (scores bruts)        │
    │      [-2.1, +3.4]   (un score par classe, pas des probabilités !)  │
    │ 3. SOFTMAX : les logits deviennent des probabilités (somme = 1)    │
    │      [-2.1, +3.4]  ->  [0.004, 0.996]                              │
    │ 4. ARGMAX : on prend la classe la plus probable                    │
    │      classe 1 = "positif", confiance = 0.996                       │
    └──────────────────────────────────────────────────────────────────┘

COMMENT LANCER (depuis la racine du projet) :
    # Un avis passé en argument :
    python scripts/predict.py "Ce film est un chef-d'œuvre absolu !"

    # Avec le détail des étapes (tokens, logits, probabilités) :
    python scripts/predict.py "Ce film est nul" --verbose

    # Mode interactif : taper des avis l'un après l'autre ('q' pour quitter) :
    python scripts/predict.py
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import argparse            # Lire le texte et les options en ligne de commande
import sys                 # Pour quitter proprement avec un message d'erreur
from pathlib import Path   # Chemins de fichiers portables (Windows/Linux)

import torch               # PyTorch : exécute le modèle
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─── CONSTANTES ─────────────────────────────────────────────────────────────

# Chemin du modèle fine-tuné, calculé comme dans train.py :
# ce fichier est dans scripts/, donc la racine du projet = 2 niveaux au-dessus.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "sentiment_model"

# DOIT être identique à la valeur utilisée à l'entraînement : le modèle a
# appris avec des avis tronqués à 256 tokens, on prédit dans les mêmes conditions.
MAX_LENGTH = 256


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du modèle et du tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    """Recharge depuis le disque le modèle fine-tuné ET son tokenizer.

    POURQUOI LES DEUX ? Ils sont indissociables : le tokenizer convertit le
    texte en numéros, et chaque numéro correspond à un embedding (vecteur)
    précis dans le modèle. Avec un autre tokenizer, le numéro 1621 ne
    voudrait plus dire "film" -> prédictions absurdes.
    """
    # Vérification avec message clair : le cas le plus fréquent d'erreur est
    # d'avoir oublié de lancer l'entraînement d'abord.
    if not MODEL_DIR.exists():
        sys.exit(
            f"Erreur : modèle introuvable dans {MODEL_DIR}\n"
            "Lancez d'abord l'entraînement : python scripts/train.py"
        )

    print(f"Chargement du modèle depuis {MODEL_DIR} ...")
    # from_pretrained lit les fichiers sauvegardés par train.py :
    #   - tokenizer : vocabulaire + règles de découpage
    #   - modèle    : architecture (config.json) + poids (model.safetensors)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    # .eval() = mode ÉVALUATION. Désactive le dropout : pendant
    # l'entraînement, le dropout "éteint" aléatoirement des neurones pour
    # régulariser ; en prédiction, on veut un résultat stable et déterministe
    # (le même avis doit toujours donner la même prédiction).
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# La prédiction proprement dite : les 4 étapes
# ─────────────────────────────────────────────────────────────────────────────
def predict_sentiment(text: str, model, tokenizer, verbose: bool = False) -> dict:
    """Prédit le sentiment d'un texte. Renvoie label + confiance + détail.

    Chaque étape correspond au schéma du haut du fichier.
    """
    # ── ÉTAPE 1 : TOKENISATION ──────────────────────────────────────────────
    # Le texte devient un dictionnaire de tenseurs :
    #   input_ids      : les numéros des tokens, ex [[5, 149, 1621, ..., 6]]
    #   attention_mask : 1 = vrai token, 0 = remplissage (pas de 0 ici car
    #                    un seul texte, donc pas de padding nécessaire)
    # return_tensors="pt" -> tenseurs PyTorch (et pas des listes Python).
    # truncation=True     -> coupe à 256 tokens si l'avis est trop long.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    if verbose:
        # convert_ids_to_tokens fait la conversion inverse, pour VOIR le découpage
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"  Tokens ({len(tokens)}) : {tokens[:15]}{'...' if len(tokens) > 15 else ''}")

    # ── ÉTAPE 2 : FORWARD PASS (passage avant dans le réseau) ───────────────
    # torch.no_grad() : on dit à PyTorch de NE PAS préparer le calcul des
    # gradients. Les gradients ne servent qu'à l'entraînement ; ici on ne
    # fait que prédire -> plus rapide et 2x moins de mémoire.
    with torch.no_grad():
        outputs = model(**inputs)   # le ** déplie le dict en arguments nommés

    # outputs.logits a la forme (1, 2) : 1 texte, 2 scores.
    # [0] enlève la dimension du batch -> tenseur [score_négatif, score_positif].
    # ATTENTION : les logits ne sont PAS des probabilités. Ils peuvent être
    # négatifs, supérieurs à 1, et ne somment pas à 1. Ce sont des scores bruts.
    logits = outputs.logits[0]

    # ── ÉTAPE 3 : SOFTMAX ───────────────────────────────────────────────────
    # La fonction softmax transforme des scores bruts en probabilités :
    #     softmax(z_i) = exp(z_i) / somme_j(exp(z_j))
    # Propriétés : toutes les valeurs sont entre 0 et 1, et leur somme = 1.
    # Exemple : [-2.1, +3.4] -> [0.004, 0.996]
    probabilities = torch.softmax(logits, dim=-1)

    # ── ÉTAPE 4 : ARGMAX + CONFIANCE ────────────────────────────────────────
    # argmax = l'indice de la plus grande probabilité = la classe prédite.
    predicted_class_id = int(torch.argmax(probabilities))
    # La "confiance" = la probabilité que le modèle attribue à cette classe.
    # NB : ce n'est pas une garantie statistique (softmax est souvent
    # sur-confiant), mais une proba proche de 0.5 signale bien un avis ambigu.
    confidence = float(probabilities[predicted_class_id])

    # id2label a été enregistré dans la config du modèle par train.py :
    # {0: "négatif", 1: "positif"} -> on récupère le nom lisible de la classe.
    label = model.config.id2label[predicted_class_id]

    if verbose:
        print(f"  Logits bruts           : négatif={logits[0]:.3f}, positif={logits[1]:.3f}")
        print(f"  Probabilités (softmax) : négatif={probabilities[0]:.4f}, "
              f"positif={probabilities[1]:.4f}")

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "négatif": round(float(probabilities[0]), 4),
            "positif": round(float(probabilities[1]), 4),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Affichage et interface en ligne de commande
# ─────────────────────────────────────────────────────────────────────────────
def print_prediction(text: str, prediction: dict):
    """Affichage formaté d'une prédiction (tronque les avis très longs)."""
    emoji = "😊" if prediction["label"] == "positif" else "😞"
    print(f"\n  Avis      : {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"  Sentiment : {emoji} {prediction['label'].upper()}")
    print(f"  Confiance : {prediction['confidence']:.2%}")


def interactive_mode(model, tokenizer):
    """Boucle interactive : l'utilisateur tape des avis, 'q' pour quitter.

    Pratique pour tester rapidement plusieurs avis sans recharger le modèle
    à chaque fois (le chargement prend quelques secondes, la prédiction
    quelques dizaines de millisecondes).
    """
    print("\nMode interactif — tapez un avis puis Entrée ('q' pour quitter)\n")
    while True:
        try:
            text = input("Votre avis > ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C ou Ctrl+D : on sort proprement au lieu de planter
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            print("  (texte vide, réessayez)")
            continue
        # verbose=True en interactif : on montre toutes les étapes, c'est le but
        prediction = predict_sentiment(text, model, tokenizer, verbose=True)
        print_prediction(text, prediction)
        print()
    print("Au revoir !")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse de sentiment d'un avis en français"
    )
    # nargs="?" : l'argument est OPTIONNEL. S'il est absent -> mode interactif.
    parser.add_argument("text", nargs="?", default=None,
                        help="L'avis à analyser (entre guillemets). "
                             "Sans argument : mode interactif.")
    parser.add_argument("--verbose", action="store_true",
                        help="Affiche les tokens, logits et probabilités détaillés")
    args = parser.parse_args()

    # Le modèle est chargé UNE fois, puis réutilisé pour toutes les prédictions
    model, tokenizer = load_model_and_tokenizer()

    if args.text is None:
        interactive_mode(model, tokenizer)
    else:
        text = args.text.strip()
        if not text:
            sys.exit("Erreur : le texte est vide.")
        prediction = predict_sentiment(text, model, tokenizer, verbose=args.verbose)
        print_prediction(text, prediction)


# Point d'entrée : exécuté seulement si on lance ce fichier directement
if __name__ == "__main__":
    main()
