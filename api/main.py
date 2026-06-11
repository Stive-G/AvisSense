"""FastAPI service for the AvisSense sentiment model."""

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
    """Load the model once at startup and release it at shutdown."""
    logger.info("Chargement du modele : %s ...", analyzer.model_id)
    start_time = time.perf_counter()
    analyzer.load()
    logger.info("Modele charge en %.1f s - API prete.", time.perf_counter() - start_time)
    yield
    analyzer.unload()
    logger.info("Ressources liberees, arret de l'API.")


app = FastAPI(
    title="AvisSense - Analyse de sentiment d'avis cinema en francais",
    description=(
        "DistilCamemBERT fine-tune sur le dataset Allocine. "
        "Classification binaire positif/negatif avec score de confiance."
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
    """Request body accepted by POST /predict."""

    text: str = Field(
        ...,
        description="L'avis en francais a analyser",
        examples=["Ce film est incroyable"],
    )


class PredictionOutput(BaseModel):
    """Prediction returned by POST /predict."""

    label: str = Field(description='"positif" ou "negatif"')
    confidence: float = Field(description="Probabilite de la classe predite")
    probabilities: dict[str, float] = Field(description="Probabilite de chaque classe")
    processing_time_ms: float = Field(description="Duree de l'inference en millisecondes")


@app.get("/info", tags=["info"])
def api_info():
    return {
        "name": "AvisSense API",
        "version": "1.1.0",
        "description": "Analyse de sentiment d'avis cinema en francais",
        "endpoints": {
            "GET /": "front minimal",
            "GET /info": "informations de l'API",
            "GET /health": "etat de l'API et du modele",
            "GET /docs": "documentation Swagger",
            "POST /predict": 'body {"text": "..."} -> label + confiance',
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
        "frontend": "Deploy the React frontend separately on Vercel.",
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
        raise HTTPException(status_code=400, detail="Le texte ne peut pas etre vide.")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Texte trop long ({len(text)} caracteres, max {MAX_INPUT_CHARS}).",
        )
    if not analyzer.is_loaded:
        raise HTTPException(status_code=503, detail="Modele en cours de chargement.")

    start_time = time.perf_counter()
    result = analyzer.predict(text)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
    logger.info(
        'Prediction : "%s..." -> %s (%.0f %%) en %.0f ms',
        text[:40],
        result["label"],
        result["confidence"] * 100,
        elapsed_ms,
    )
    return PredictionOutput(**result, processing_time_ms=elapsed_ms)
