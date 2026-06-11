"""Service FastAPI pour l'analyse de sentiment AvisSense."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import SentimentAnalyzer
from src.utils import clean_text

MAX_INPUT_CHARS = 5000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("avissense")

analyzer = SentimentAnalyzer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle est chargé une seule fois au démarrage pour éviter les rechargements par requête.
    logger.info("Chargement du modèle : %s ...", analyzer.model_id)
    start_time = time.perf_counter()
    analyzer.load()
    logger.info("Modèle chargé en %.1f s - API prête.", time.perf_counter() - start_time)
    yield
    analyzer.unload()
    logger.info("Ressources libérées, arrêt de l'API.")


app = FastAPI(
    title="AvisSense - Analyse de sentiment d'avis cinéma en français",
    description=(
        "DistilCamemBERT fine-tuné sur le dataset Allociné. "
        "Classification binaire positif/négatif avec score de confiance."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewInput(BaseModel):
    """Corps accepté par POST /predict."""

    text: str = Field(
        ...,
        description="Avis en français à analyser",
        examples=["Ce film est incroyable"],
    )


class PredictionOutput(BaseModel):
    """Réponse renvoyée par POST /predict."""

    label: str = Field(description='"positif" ou "negatif"')
    confidence: float = Field(description="Probabilité de la classe prédite")
    probabilities: dict[str, float] = Field(description="Probabilité de chaque classe")
    processing_time_ms: float = Field(description="Durée de l'inférence en millisecondes")


@app.get("/info", tags=["info"])
def api_info():
    return {
        "name": "AvisSense API",
        "version": "1.1.0",
        "description": "Analyse de sentiment d'avis cinéma en français",
        "endpoints": {
            "GET /": "informations générales",
            "GET /info": "informations de l'API",
            "GET /health": "état du service et du modèle",
            "GET /docs": "documentation interactive",
            "POST /predict": 'analyse d\'un avis via {"text": "..."}',
        },
    }


@app.get("/", tags=["info"])
def api_root():
    return {
        "name": "AvisSense API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "frontend": "Le frontend React est déployé séparément sur Vercel.",
    }


@app.get("/health", tags=["monitoring"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": analyzer.is_loaded,
        "model_id": analyzer.model_id,
        "device": analyzer.device,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict_sentiment(review: ReviewInput):
    text = clean_text(review.text)
    if not text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Texte trop long ({len(text)} caractères, maximum {MAX_INPUT_CHARS}).",
        )
    if not analyzer.is_loaded:
        raise HTTPException(status_code=503, detail="Le modèle est en cours de chargement.")

    # Le temps de réponse est renvoyé au front pour l'affichage utilisateur.
    start_time = time.perf_counter()
    result = analyzer.predict(text)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
    logger.info(
        'Prédiction : "%s..." -> %s (%.0f %%) en %.0f ms',
        text[:40],
        result["label"],
        result["confidence"] * 100,
        elapsed_ms,
    )
    return PredictionOutput(**result, processing_time_ms=elapsed_ms)
