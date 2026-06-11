"""
push_model_to_hub.py — Envoie le modèle fine-tuné sur le Hugging Face Hub.
===========================================================================

POURQUOI CE SCRIPT EXISTE :
    Le modèle entraîné pèse ~270 Mo. GitHub refuse les fichiers de plus de
    100 Mo : impossible de versionner les poids avec le code. La solution
    standard de l'écosystème ML :

        CODE   -> GitHub          (léger, versionné, revu en PR)
        POIDS  -> Hugging Face Hub (stockage de modèles, gratuit, versionné aussi)

    Au déploiement, le Space lit la variable d'environnement MODEL_ID
    (ex: "rima/avissense-distilcamembert") et télécharge les poids depuis
    le Hub au démarrage de l'API. Code et poids voyagent séparément mais
    se retrouvent en production.

PRÉREQUIS (une seule fois) :
    1. Créer un compte sur https://huggingface.co
    2. Créer un token d'accès : Settings > Access Tokens > type "Write"
       (le type "Read" ne suffit pas : on veut ÉCRIRE sur le Hub)
    3. Se connecter en local :  huggingface-cli login   (coller le token)

COMMENT LANCER (depuis la racine du projet) :
    python scripts/push_model_to_hub.py --repo VOTRE_PSEUDO/avissense-distilcamembert
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import argparse            # Lire le nom du repo en ligne de commande
import sys                 # Sortie propre en cas d'erreur
from pathlib import Path   # Chemins portables

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Même convention de chemins que les autres scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model" / "sentiment_model"


def main():
    parser = argparse.ArgumentParser(description="Pousse le modèle fine-tuné sur le HF Hub")
    # required=True : pas de valeur par défaut volontairement — le nom du
    # repo contient VOTRE pseudo, on ne peut pas le deviner.
    parser.add_argument("--repo", required=True,
                        help="Nom du repo Hub, ex : rima/avissense-distilcamembert")
    args = parser.parse_args()

    # Vérification : le modèle doit avoir été entraîné avant d'être publié
    if not MODEL_DIR.exists():
        sys.exit(f"Erreur : modèle introuvable dans {MODEL_DIR}. Lancez d'abord train.py.")

    # On recharge le modèle ET le tokenizer : les deux doivent être publiés
    # ensemble (le Space aura besoin des deux pour faire des prédictions).
    print(f"Chargement du modèle local depuis {MODEL_DIR} ...")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    # push_to_hub :
    #   - crée le repo sur le Hub s'il n'existe pas encore,
    #   - uploade les poids (~270 Mo, peut prendre quelques minutes),
    #   - utilise le token enregistré par `huggingface-cli login`.
    print(f"Upload vers https://huggingface.co/{args.repo} ...")
    model.push_to_hub(args.repo)
    tokenizer.push_to_hub(args.repo)
    print("Terminé ! Le modèle est en ligne sur le Hub.")
    print(f"Sur le Space, réglez la variable :  MODEL_ID = {args.repo}")


if __name__ == "__main__":
    main()
