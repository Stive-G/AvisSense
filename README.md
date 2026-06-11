# AvisSense

AvisSense est un projet de fin de module M106 visant à construire une application d'analyse intelligente du sentiment d'avis en français.

## Objectif

Préparer une base professionnelle pour une application capable d'analyser le sentiment d'un avis en français, avec une interface Gradio et une future intégration de modèles Transformers.

## Stack technique

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Gradio
- scikit-learn
- pandas / numpy
- Hugging Face Spaces

## Arborescence

```text
avis-sense/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── predict.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   └── avis_sense_training.ipynb
├── model/
│   └── README.md
└── docs/
    └── project_plan.md
```

## Installation

```bash
uv init
uv add gradio torch transformers datasets scikit-learn pandas numpy accelerate sentencepiece protobuf
```

## Lancement local

```bash
uv run python app.py
```

## Prochaines étapes

- Tester une baseline avec un modèle pré-entraîné.
- Brancher l'interface Gradio sur une vraie fonction de prédiction.
- Charger le dataset Allociné via Hugging Face Datasets.
- Préparer un fine-tuning léger.
- Évaluer le modèle avec accuracy, precision, recall et f1-score.
- Déployer l'application sur Hugging Face Spaces.
