"""
main.py — API FastAPI d'analyse de sentiment (AvisSense).
==========================================================

API REST pure : le frontend est développé séparément par un autre
développeur et consommera ces endpoints (CORS activé pour ça).

Endpoints :
    GET  /          -> informations sur l'API (nom, version, endpoints)
    GET  /health    -> état de l'API + modèle chargé (vérification déploiement)
    POST /predict   -> {"text": "Ce film est incroyable"}
                    -> {"label": "positif", "confidence": 0.94,
                        "probabilities": {"négatif": 0.06, "positif": 0.94},
                        "processing_time_ms": 45.2}

Points d'architecture importants :
    - Le modèle est chargé UNE SEULE FOIS au démarrage du serveur (lifespan).
      Le charger à chaque requête prendrait plusieurs secondes par prédiction.
    - L'inférence est faite "à la main" (tokenizer -> logits -> softmax),
      comme dans scripts/predict.py : on contrôle chaque étape.
    - MODEL_ID est configurable par variable d'environnement : en local c'est
      le dossier model/sentiment_model/, sur Hugging Face Spaces c'est le repo
      Hub du modèle (ex: "rima/avissense-distilcamembert") téléchargé au boot.

Lancement local (depuis la racine du projet) :
    uvicorn api.main:app --reload
Puis ouvrir :
    http://127.0.0.1:8000/docs   -> documentation Swagger interactive
                                    (contrat d'API pour le dev frontend)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Chemin local par défaut ; remplacé par un repo HF Hub sur Spaces
# (variable d'environnement MODEL_ID dans les Settings du Space).
MODEL_ID = os.getenv("MODEL_ID", str(PROJECT_ROOT / "model" / "sentiment_model"))

# Doit correspondre à la longueur utilisée à l'entraînement (train.py)
MAX_LENGTH = 256

# Limite de taille de l'entrée : le modèle tronque à 256 tokens de toute
# façon, mais on rejette les entrées absurdement longues avant traitement.
MAX_INPUT_CHARS = 5000

# Logger : trace le démarrage et chaque prédiction dans la console du serveur
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("avissense")

# Conteneur global rempli au démarrage par le lifespan (modèle + tokenizer)
ml_resources = {}


# ---------------------------------------------------------------------------
# Cycle de vie : chargement du modèle au démarrage
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Code exécuté au démarrage (avant `yield`) et à l'arrêt (après).

    Charger le modèle ici garantit qu'il est en mémoire AVANT la première
    requête : toutes les prédictions sont ensuite rapides (~50 ms sur CPU).
    """
    logger.info("Chargement du modèle : %s ...", MODEL_ID)
    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()  # Mode évaluation : désactive le dropout -> prédictions stables

    ml_resources["tokenizer"] = tokenizer
    ml_resources["model"] = model
    logger.info("Modèle chargé en %.1f s — API prête.", time.perf_counter() - start_time)

    yield  # ----- l'API tourne ici -----

    ml_resources.clear()  # Libère la mémoire à l'arrêt du serveur
    logger.info("Ressources libérées, arrêt de l'API.")


app = FastAPI(
    title="AvisSense — Analyse de sentiment d'avis cinéma en français",
    description="DistilCamemBERT fine-tuné sur le dataset Allociné. "
                "Classification binaire positif/négatif avec score de confiance.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS : indispensable ici — le frontend est développé et hébergé séparément,
# le navigateur bloquerait ses appels vers l'API sans ces en-têtes.
# En production, restreindre allow_origins au domaine réel du frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schémas d'entrée / sortie — validés automatiquement par Pydantic
# ---------------------------------------------------------------------------
class ReviewInput(BaseModel):
    """Corps attendu de la requête POST /predict."""
    text: str = Field(
        ...,
        description="L'avis en français à analyser",
        examples=["Ce film est incroyable"],
    )


class PredictionOutput(BaseModel):
    """Réponse renvoyée par POST /predict."""
    label: str = Field(description='"positif" ou "négatif"')
    confidence: float = Field(description="Probabilité de la classe prédite (0 à 1)")
    probabilities: dict[str, float] = Field(
        description="Probabilité de chaque classe (somme = 1)"
    )
    processing_time_ms: float = Field(description="Durée de l'inférence en millisecondes")


# ---------------------------------------------------------------------------
# Fonction d'inférence (mêmes 4 étapes que scripts/predict.py)
# ---------------------------------------------------------------------------
def run_inference(text: str) -> dict:
    """Tokenise le texte, exécute le modèle et renvoie label + probabilités."""
    tokenizer = ml_resources["tokenizer"]
    model = ml_resources["model"]

    # Étape 1 : texte -> tenseurs d'identifiants de tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    # Étape 2 : forward pass sans gradients (inférence pure)
    with torch.no_grad():
        logits = model(**inputs).logits[0]

    # Étape 3 : logits -> probabilités (softmax)
    probabilities = torch.softmax(logits, dim=-1)

    # Étape 4 : classe la plus probable + son score de confiance
    predicted_class_id = int(torch.argmax(probabilities))
    label = model.config.id2label[predicted_class_id]  # "positif" / "négatif"

    return {
        "label": label,
        "confidence": round(float(probabilities[predicted_class_id]), 4),
        "probabilities": {
            model.config.id2label[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["info"])
def api_info():
    """Point d'entrée : décrit l'API (utile au dev frontend et aux tests)."""
    return {
        "name": "AvisSense API",
        "version": "1.0.0",
        "description": "Analyse de sentiment d'avis cinéma en français "
                       "(DistilCamemBERT fine-tuné sur Allociné)",
        "endpoints": {
            "GET /": "ces informations",
            "GET /health": "état de l'API et du modèle",
            "GET /docs": "documentation interactive Swagger",
            "POST /predict": 'body {"text": "..."} -> label + confiance',
        },
    }


@app.get("/health", tags=["monitoring"])
def health_check():
    """Vérifie que l'API tourne et que le modèle est bien chargé en mémoire."""
    return {
        "status": "ok",
        "model_loaded": "model" in ml_resources,
        "model_id": MODEL_ID,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict_sentiment(review: ReviewInput):
    """Prédit le sentiment (positif/négatif) d'un avis en français.

    Erreurs gérées :
        400 — texte vide ou composé uniquement d'espaces
        400 — texte dépassant 5000 caractères
        422 — JSON malformé ou champ "text" manquant (automatique, Pydantic)
        503 — modèle pas encore chargé
    """
    text = review.text.strip()

    # --- Validation de l'entrée ----------------------------------------------
    if not text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Texte trop long ({len(text)} caractères, max {MAX_INPUT_CHARS}).",
        )
    if "model" not in ml_resources:
        raise HTTPException(status_code=503, detail="Modèle en cours de chargement.")

    # --- Prédiction + mesure du temps de traitement ---------------------------
    start_time = time.perf_counter()
    result = run_inference(text)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    logger.info('Prédiction : "%s..." -> %s (%.0f %%) en %.0f ms',
                text[:40], result["label"], result["confidence"] * 100, elapsed_ms)

    return PredictionOutput(**result, processing_time_ms=elapsed_ms)
