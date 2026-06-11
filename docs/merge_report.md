# Rapport de fusion intelligente

## Base retenue

La branche `dev` est creee depuis `main`. La branche `main` reste la reference
fonctionnelle et n'a pas ete modifiee.

## Fichiers conserves depuis main

- `scripts/train.py` : fine-tuning DistilCamemBERT sur Allocine.
- `scripts/evaluate.py` : evaluation, metriques et analyse des erreurs.
- `scripts/push_model_to_hub.py` : publication du modele.
- `api/main.py` : contrat FastAPI et front statique, avec inference mutualisee.
- `frontend/` : interface web servie par FastAPI.
- `Dockerfile` : deploiement Docker Hugging Face Spaces.
- `requirements.txt` : dependances de main, completees avec Gradio.
- `README.md` : documentation principale de main.

## Elements recuperes depuis test_russ

- `app.py` : interface Gradio, reecrite pour utiliser le modele entraine par main.
- `src/utils.py` : utilitaires de nettoyage et de presentation, adaptes aux labels
  `positif` et `negatif`.
- `src/__init__.py` : package applicatif partage.

## Conflits resolus

- Les squelettes `src/train.py`, `src/evaluate.py` et `src/predict.py` de
  `test_russ` n'ont pas ete repris.
- Le `README.md`, le `Dockerfile`, l'API et les scripts ML de `main` restent
  prioritaires.
- Le chemin `model/saved_model` de `test_russ` a ete remplace par la convention
  de main : `model/sentiment_model`, ou la variable d'environnement `MODEL_ID`.
- Le chargement CamemBERT specifique de `test_russ` a ete remplace par les
  classes Auto de Transformers, compatibles avec DistilCamemBERT.

## Ameliorations effectuees

- Ajout de `src/inference.py`, source unique pour charger le modele et predire.
- FastAPI, Gradio et le CLI utilisent la meme logique d'inference.
- Chargement unique du modele par processus.
- Gradio charge le modele a la premiere prediction.
- Le CLI conserve les modes argument, interactif et verbose.
- Ajout de Gradio aux dependances.

## Architecture finale

```text
api/main.py                 FastAPI + front statique
app.py                      interface Gradio optionnelle
frontend/                   interface HTML/CSS/JS
scripts/train.py            entrainement
scripts/evaluate.py         evaluation
scripts/predict.py          prediction CLI
scripts/push_model_to_hub.py
src/inference.py            chargement et inference mutualises
src/utils.py                nettoyage et formatage d'affichage
model/sentiment_model/      modele entraine, non versionne
```
