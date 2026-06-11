---
title: AvisSense
emoji: "🎬"
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Analyse de sentiment d'avis cinema en francais avec DistilCamemBERT.
---
# 🎬 AvisSense — Analyse de sentiment d'avis cinéma (français)

> Projet de fin de module **M106 — Intro ML/DL** · Sujet : analyse de sentiment d'avis en français (secteur culture)

Classification binaire d'avis de cinéma : **on colle un avis en français, le modèle prédit positif ou négatif avec un score de confiance** (la consigne du sujet, mot pour mot). Un **DistilCamemBERT** est fine-tuné (transfer learning) sur le dataset **Allociné**, **servi par une API FastAPI** avec un **petit front** intégré, le tout déployé sur **Hugging Face Spaces** — toute la chaîne, sans usine à gaz.

> Le repo inclut un **front minimal** (HTML/CSS/JS, servi par l'API à `GET /`) qui garantit la chaîne complète déployée. Un front plus complet est développé séparément par un autre membre de l'équipe : il consommera les mêmes endpoints (contrat d'API en section 8, Swagger sur `/docs`, CORS déjà activé).

```
POST /predict  {"text": "Ce film est incroyable"}
            -> {"label": "positif", "confidence": 0.94, ...}
```

---

## 1. Objectif

- Fine-tuner un modèle de langage pré-entraîné (transfer learning) pour classer des avis de cinéma en français : positif / négatif.
- Trancher la **décision architect** : geler le modèle et n'entraîner qu'une tête, ou fine-tuner entièrement ? (section 4)
- **Servir** le modèle via une API REST (FastAPI) + un petit front « colle un avis → résultat ».
- Déployer publiquement la chaîne complète sur Hugging Face Spaces.

### Conformité aux consignes du module

| Consigne | Où c'est respecté |
|---|---|
| Orienté Deep Learning (fine-tuning d'un modèle de texte) | `scripts/train.py` — DistilCamemBERT fine-tuné sur Allociné |
| Modèle **servi** avec FastAPI | `api/main.py` — `POST /predict` |
| **Petit front** | `frontend/` — servi par l'API à `GET /` |
| Déployé (HF Spaces ou équivalent) | `Dockerfile` — Space Docker, section 9 |
| Secteur ≠ démos du cours | culture / cinéma |
| Décisions défendues (modèle, stratégie de transfer) | sections 3 et 4 — les **deux** stratégies sont implémentées et comparées |
| Livrables | code + README (ce fichier) + lien du modèle déployé (section 9) + vidéo ≤ 10 min sur Teams **avant vendredi 13h** |

## 2. Dataset — Allociné (avis ciné uniquement)

- **Source** : [`allocine`](https://huggingface.co/datasets/allocine) sur le Hugging Face Hub.
- **Contenu** : ~200 000 avis de films en français, étiquetés `0` (négatif) ou `1` (positif), labels dérivés des notes utilisateurs (donc fiables). Classes équilibrées ~50/50.
- **Splits fournis** : train (160k) / validation (20k) / test (20k) — déjà séparés, aucun risque de fuite de données.
- **Colonnes** : `review` (texte) et `label`.
- **Sous-échantillonnage** : entraînement sur **8 000 avis** (validation 2 000, test 2 000) pour rester rapide (~20 min sur GPU Colab gratuit). Le modèle pré-entraîné connaît déjà le français : il n'apprend que la frontière positif/négatif, et la performance sature vite — les 160k complets n'apporteraient que 1-2 points de F1 pour 20× plus de calcul. Ajustable via `--max-train`.

## 3. Modèle — DistilCamemBERT

| Modèle | Langue | Paramètres | Vitesse | Performance FR |
|---|---|---|---|---|
| CamemBERT-base | Français | 110M | 1× | Excellente |
| **DistilCamemBERT** ✅ | Français | 68M | ~2× | ~97 % de CamemBERT |
| DistilBERT multilingual | 104 langues | 134M | ~1.5× | Moyenne (capacité diluée) |

**Choix : [`cmarkea/distilcamembert-base`](https://huggingface.co/cmarkea/distilcamembert-base)** — spécialisé français (un modèle multilingue dilue sa capacité sur 104 langues et tokenise moins bien le français), 2× plus rapide que CamemBERT à l'entraînement comme en inférence, pour une perte négligeable sur une tâche binaire. Crucial pour servir le modèle sur le CPU gratuit de Spaces.

## 4. Décision Architect — geler le backbone ou fine-tuner ?

C'est l'arbitrage central du projet, et **les deux stratégies sont implémentées** dans `train.py` pour le trancher avec des chiffres plutôt qu'une intuition :

| | Stratégie A — fine-tuning complet (défaut) | Stratégie B — backbone gelé (`--freeze-backbone`) |
|---|---|---|
| Paramètres entraînés | 68M (100 %) | ~0.6M (~1 %, la tête seule) |
| Learning rate | 2e-5 (faible : on ajuste sans détruire) | 1e-3 (la tête part de zéro) |
| Coût d'entraînement | ~20 min (GPU Colab) | ~3× plus rapide, envisageable sur CPU |
| F1 attendu (test) | **≈ 0.93–0.96** | ≈ 0.85–0.89 |

```bash
# Stratégie A — fine-tuning complet
python scripts/train.py

# Stratégie B — backbone gelé, tête seule
python scripts/train.py --freeze-backbone --learning-rate 1e-3
```

Chaque run écrit ses métriques dans `model/sentiment_model/training_metrics.json` (avec le champ `strategy`) : comparer les deux JSON donne l'arbitrage chiffré.

**Décision retenue : fine-tuning complet.** Le gel du backbone utilise les features génériques du pré-entraînement (masked language modeling), qui ne sont pas optimisées pour discriminer le sentiment ; laisser le backbone s'adapter au vocabulaire des critiques de cinéma gagne ~5-8 points de F1. Le surcoût de calcul (20 min de Colab gratuit) est négligeable, et le coût d'**inférence est identique** dans les deux cas — l'argument "économie" du gel ne vaut que pour l'entraînement. Le gel resterait pertinent avec très peu de données (< 1 000 exemples, risque d'overfitting) ou sans aucun GPU disponible — ce n'est pas notre cas.

## 5. Architecture du projet

```
AvisSense/
├── api/
│   └── main.py              # API FastAPI : front sur /, /info, /health, POST /predict
├── frontend/
│   ├── index.html           # Front minimal : zone de texte + bouton + résultat
│   ├── style.css            # Style (label coloré + barre de confiance)
│   └── script.js            # fetch() vers POST /predict
├── model/
│   └── sentiment_model/     # Modèle fine-tuné (créé par train.py, non versionné)
├── scripts/
│   ├── train.py             # Entraînement : les 2 stratégies (gel / fine-tuning)
│   ├── predict.py           # Prédiction en CLI (inférence manuelle détaillée)
│   ├── evaluate.py          # Évaluation test set + analyse des erreurs
│   └── push_model_to_hub.py # Upload du modèle sur le HF Hub
├── Dockerfile               # Déploiement Hugging Face Spaces (Docker)
├── requirements.txt         # Dépendances Python
├── .gitignore
└── README.md
```

**Rôle des dossiers** :
- `scripts/` — cycle de vie du modèle (entraînement, évaluation, publication). Séparé de l'API : on peut ré-entraîner sans toucher au serveur.
- `api/` — le serveur. Charge le modèle une fois au démarrage (lifespan), expose les endpoints, sert le front, CORS activé pour le futur front externe.
- `frontend/` — le petit front demandé par les consignes : fichiers statiques servis par FastAPI, aucune dépendance supplémentaire. Remplaçable par le front complet de l'équipe sans toucher à l'API.
- `model/` — artefact produit par l'entraînement. Ignoré par git (270 Mo) : les poids sont stockés sur le HF Hub, le code sur GitHub.

## 6. Installation

```bash
git clone https://github.com/VOTRE_PSEUDO/AvisSense.git
cd AvisSense

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

## 7. Entraînement et évaluation

```bash
# Entraînement standard (fine-tuning complet, 8000 avis, 2 époques)
python scripts/train.py

# Comparaison avec le backbone gelé (décision architect, section 4)
python scripts/train.py --freeze-backbone --learning-rate 1e-3

# Version rapide pour tester le pipeline sur CPU (~10 min)
python scripts/train.py --max-train 2000 --max-val 500 --max-test 500 --epochs 1
```

Sur **Google Colab** (GPU gratuit, recommandé) : uploader le dossier, puis :
```python
!pip install -q transformers datasets accelerate scikit-learn
!python scripts/train.py
```

Le script affiche l'exploration du dataset, la progression, puis les métriques finales sur le **test set** avec matrice de confusion, et sauvegarde le modèle dans `model/sentiment_model/`.

```bash
# Prédiction en CLI (--verbose : tokens, logits, softmax ; sans argument : mode interactif)
python scripts/predict.py "Ce film est un chef-d'œuvre absolu !"

# Évaluation détaillée : rapport par classe, matrice de confusion,
# et les erreurs où le modèle était le plus confiant (analyse des limites)
python scripts/evaluate.py --max-test 2000 --show-errors 5
```

## 8. Lancement : API + front

```bash
uvicorn api.main:app --reload
# Front    : http://127.0.0.1:8000
# Swagger  : http://127.0.0.1:8000/docs
```

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Le front minimal : coller un avis → prédiction + confiance |
| GET | `/info` | Informations sur l'API (JSON) |
| GET | `/health` | `{"status": "ok", "model_loaded": true, ...}` |
| POST | `/predict` | Prédiction de sentiment |

**POST /predict** — requête :
```json
{ "text": "Ce film est incroyable" }
```

Réponse `200` :
```json
{
  "label": "positif",
  "confidence": 0.9812,
  "probabilities": { "négatif": 0.0188, "positif": 0.9812 },
  "processing_time_ms": 47.3
}
```

Erreurs :
| Code | Cas | Corps |
|---|---|---|
| 400 | texte vide / espaces | `{"detail": "Le texte ne peut pas être vide."}` |
| 400 | texte > 5000 caractères | `{"detail": "Texte trop long (...)"}` |
| 422 | JSON malformé ou champ `text` manquant | détail Pydantic automatique |
| 503 | modèle pas encore chargé | `{"detail": "Modèle en cours de chargement."}` |

Exemples d'appel :
```bash
# PowerShell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" -Body '{"text": "Ce film est incroyable"}'

# curl
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Ce film est incroyable"}'
```

CORS est activé (`allow_origins=["*"]`) pour que le frontend hébergé ailleurs puisse appeler l'API ; à restreindre au domaine réel en production.

## 9. Déploiement sur Hugging Face Spaces

Deux artefacts, deux destinations : **les poids sur le Hub**, **l'API sur un Space Docker** (qui télécharge les poids au démarrage).

### Étape A — Pousser le modèle sur le Hub
```bash
huggingface-cli login        # token "Write" depuis Settings > Access Tokens
python scripts/push_model_to_hub.py --repo VOTRE_PSEUDO/avissense-distilcamembert
```

### Étape B — Créer le Space
1. https://huggingface.co/new-space : SDK **Docker**, hardware **CPU basic (gratuit)**.
2. Dans **Settings > Variables** du Space : `MODEL_ID = VOTRE_PSEUDO/avissense-distilcamembert`
3. Pousser le code :
```bash
git remote add space https://huggingface.co/spaces/VOTRE_PSEUDO/avissense
git push space main
```

La chaîne complète est alors visible à l'URL publique du Space : `https://VOTRE_PSEUDO-avissense.hf.space/` ouvre directement le front « colle un avis → résultat » (et `/docs` la documentation de l'API). **C'est ce lien qu'on remet comme livrable « modèle déployé ».** La première requête après un réveil du Space est lente (téléchargement + chargement du modèle) ; `/health` permet de vérifier que le modèle est prêt — à faire avant la démo vidéo.

## 10. Limites

- **Domaine ciné** : entraîné sur des avis de films — les performances baissent hors domaine (produits, restaurants) car le vocabulaire diffère (« lourd » est négatif pour un film, neutre pour un plat).
- **Binaire** : pas de classe « neutre » — un avis mitigé est forcé dans un camp, généralement avec une confiance plus faible (~0.5-0.7).
- **Ironie / sarcasme** : « Bravo, 2 h de ma vie perdues » contient des mots positifs avec un sens négatif.
- **Confiance non calibrée** : le score softmax est souvent sur-confiant ; 0.94 ne signifie pas « 94 % de chances d'avoir raison » au sens statistique.
- **Troncature à 256 tokens** : la fin des très longs avis est ignorée.

## 11. Améliorations futures

- Classe **neutre** (3 classes, autre dataset).
- Entraînement sur les **160k avis** complets (+1-2 points de F1).
- **Calibration** du score de confiance (temperature scaling).
- Endpoint **batch** (`POST /predict/batch`) pour analyser plusieurs avis en un appel.
- Export **ONNX + quantization int8** : inférence 2-4× plus rapide sur CPU.
- **Monitoring** : logger les prédictions à faible confiance pour ré-annotation (amélioration continue).

## 12. Interface Gradio optionnelle

En plus de l'API FastAPI et de son front statique, une interface Gradio est
disponible :

```bash
python app.py
```

FastAPI, Gradio et le script de prediction utilisent tous `src/inference.py`.
Le modele est charge depuis `model/sentiment_model/` en local, ou depuis le
repo Hugging Face indique par la variable d'environnement `MODEL_ID`.

Le rapport de fusion des branches est disponible dans
`docs/merge_report.md`.

---

*Projet réalisé dans le cadre du module M106 — Intro ML/DL.*
