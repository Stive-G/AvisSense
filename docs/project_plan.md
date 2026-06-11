# Plan de projet AvisSense

## 1. Baseline avec modèle pré-entraîné

Tester un modèle Transformers déjà entraîné pour l'analyse de sentiment en français afin d'obtenir une première référence.

## 2. Interface Gradio

Créer une interface simple permettant de saisir un avis en français et d'afficher le sentiment prédit.

## 3. Chargement du dataset Allociné

Utiliser Hugging Face Datasets pour charger un jeu de données d'avis en français, puis préparer les splits d'entraînement, validation et test.

## 4. Fine-tuning léger

Adapter un modèle pré-entraîné au dataset Allociné avec un entraînement court et reproductible.

## 5. Évaluation

Mesurer les performances avec accuracy, precision, recall et f1-score.

## 6. Déploiement Hugging Face Spaces

Préparer les fichiers nécessaires au déploiement de l'application Gradio sur Hugging Face Spaces.

## 7. Préparation soutenance

Documenter les choix techniques, les résultats, les limites et les pistes d'amélioration.
