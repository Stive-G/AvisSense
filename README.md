---
title: AvisSense
emoji: "🎬"
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Analyse de sentiment d'avis cinéma en français.
---

# AvisSense

Projet réalisé dans le cadre du module **M106 – Introduction au Machine Learning et Deep Learning**.

## Sujet choisi

Analyse de sentiment d'avis en français à partir d'avis de cinéma.

## Objectif

L'utilisateur saisit ou colle un avis en français. L'application indique ensuite si cet avis est plutôt **positif** ou **négatif**, avec un **niveau de confiance**.

## Technologies utilisées

- Python
- PyTorch
- Hugging Face Transformers
- Dataset Allociné
- FastAPI
- Gradio
- Hugging Face Spaces

## Dataset

Le projet utilise le dataset **Allociné** disponible sur le Hugging Face Hub.  
Il contient des avis de films en français avec deux classes :

- `0` : négatif
- `1` : positif

## Modèle retenu

Le modèle utilisé est **DistilCamemBERT** (`cmarkea/distilcamembert-base`).

Ce choix permet de conserver un modèle adapté au français tout en restant plus léger qu'un CamemBERT complet, ce qui facilite l'entraînement et le déploiement.

## Principe ML / DL

Le projet repose sur le **transfer learning** :

- on part d'un modèle déjà pré-entraîné sur du texte français ;
- on réalise ensuite un **fine-tuning** sur des avis Allociné ;
- le modèle apprend ainsi à distinguer les avis positifs des avis négatifs dans le domaine du cinéma.

Deux stratégies sont prévues dans l'entraînement :

- fine-tuning complet ;
- gel du backbone avec entraînement de la tête de classification uniquement.

## Ce que fait l'utilisateur

1. Il écrit ou colle un avis de film en français.
2. Il lance l'analyse.
3. L'application affiche :
   - le sentiment prédit ;
   - le niveau de confiance ;
   - un retour simple et lisible.

## Architecture du projet

```text
AvisSense/
├── api/                    # API FastAPI
├── frontend/               # Frontend React pour Vercel
├── model/                  # Modèle entraîné en local
├── scripts/                # Entraînement, évaluation, prédiction, publication
├── src/                    # Logique partagée d'inférence et utilitaires
├── tests/                  # Tests unitaires
├── app.py                  # Interface Gradio
├── Dockerfile              # Déploiement Hugging Face Spaces
└── README.md
```

## Scripts principaux

- `python scripts/train.py`
  - entraîne le modèle et sauvegarde les poids dans `model/sentiment_model`
- `python scripts/evaluate.py`
  - évalue le modèle sur le jeu de test
- `python scripts/predict.py "Ce film est magnifique"`
  - lance une prédiction en ligne de commande
- `uvicorn api.main:app --reload`
  - démarre l'API FastAPI
- `python app.py`
  - lance l'interface Gradio
- `python scripts/push_model_to_hub.py --repo utilisateur/avissense-distilcamembert`
  - publie le modèle sur le Hugging Face Hub

## Résultats obtenus

Sur un test réduit de **500 avis**, un entraînement a produit les métriques suivantes :

- accuracy : `0.9100`
- precision : `0.9012`
- recall : `0.9125`
- f1 : `0.9068`

Matrice de confusion observée :

```text
                 |  prédit négatif |  prédit positif
----------------------------------------------------
    vrai négatif |             236 |              24
    vrai positif |              21 |             219
```

Ces résultats montrent que le modèle est déjà solide sur une configuration d'entraînement courte.

## Limites du modèle

- Le modèle est entraîné sur des **avis de cinéma** : ses performances peuvent baisser hors de ce domaine.
- La classification est **binaire** : il n'y a pas de classe neutre.
- L'ironie, le sarcasme et les avis très ambigus restent difficiles.
- Le score de confiance est utile pour l'interprétation, mais il ne garantit pas une certitude absolue.
- Les textes très longs sont tronqués à `256` tokens.

## Lancement local

Créer un environnement virtuel puis installer les dépendances :

```bash
uv venv
uv pip install -r requirements.txt
```

Exemples de commandes utiles :

```bash
python scripts/train.py --max-train 2000 --max-val 500 --max-test 500 --epochs 1
python scripts/evaluate.py --max-test 500
python scripts/predict.py "Ce film est magnifique"
uvicorn api.main:app --reload
python app.py
```

## Déploiement

### Hugging Face

Le projet peut être déployé sur **Hugging Face Spaces** :

- le code de l'application part sur le Space ;
- le modèle entraîné est publié sur le **Hugging Face Hub** ;
- la variable `MODEL_ID` permet au Space de charger le bon modèle.

### Vercel

Le frontend React peut être déployé séparément sur **Vercel** :

- `Root Directory` : `frontend`
- `Framework Preset` : `Vite`
- variable d'environnement :
  - `VITE_API_BASE_URL=https://votre-space.hf.space`

## API

Route principale de prédiction :

```text
POST /predict
```

Exemple de corps :

```json
{ "text": "Ce film est incroyable" }
```

Exemple de réponse :

```json
{
  "label": "positif",
  "confidence": 0.9812,
  "probabilities": {
    "négatif": 0.0188,
    "positif": 0.9812
  },
  "processing_time_ms": 47.3
}
```

## Remarque

`src/inference.py` charge le modèle :

- depuis `MODEL_ID` si la variable d'environnement est définie ;
- sinon depuis `model/sentiment_model` en local.
