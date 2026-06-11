"""
main.py — API FastAPI d'analyse de sentiment (AvisSense).
==========================================================

CE QUE FAIT CE FICHIER :
    Il transforme le modèle entraîné en SERVICE WEB : n'importe quelle
    application (le frontend de l'équipe, un script, curl...) peut envoyer
    un avis en HTTP et recevoir la prédiction en JSON.

    API REST pure : le frontend est développé séparément par un autre
    développeur et consommera ces endpoints (d'où le CORS activé plus bas).

LES ENDPOINTS :
    GET  /          -> informations sur l'API (nom, version, endpoints)
    GET  /health    -> état de l'API + modèle chargé (vérification déploiement)
    POST /predict   -> {"text": "Ce film est incroyable"}
                    -> {"label": "positif", "confidence": 0.94,
                        "probabilities": {"négatif": 0.06, "positif": 0.94},
                        "processing_time_ms": 45.2}

POURQUOI FASTAPI (et pas Flask) ?
    - Validation AUTOMATIQUE des entrées via Pydantic : si le JSON est
      malformé ou si "text" manque, FastAPI renvoie une erreur 422 claire
      sans qu'on écrive une ligne de code.
    - Documentation Swagger générée automatiquement sur /docs : le dev
      frontend y voit le contrat d'API et peut tester en direct.
    - Asynchrone natif, performant, standard actuel pour servir du ML.

DÉCISION D'ARCHITECTURE IMPORTANTE :
    Le modèle est chargé UNE SEULE FOIS au démarrage du serveur (mécanisme
    "lifespan" ci-dessous), PAS à chaque requête. Charger le modèle prend
    plusieurs secondes ; une prédiction, quelques dizaines de millisecondes.
    Si on chargeait à chaque requête, chaque appel prendrait 5 secondes.

COMMENT LANCER (depuis la racine du projet) :
    uvicorn api.main:app --reload
    │       │   │    │    └─ redémarre tout seul quand le code change (dev)
    │       │   │    └─ la variable `app` définie dans ce fichier
    │       │   └─ le fichier api/main.py
    │       └─ le package api/
    └─ uvicorn = le serveur web ASGI qui exécute FastAPI

    Puis ouvrir :  http://127.0.0.1:8000/docs  (documentation interactive)
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import logging                          # Tracer ce qui se passe côté serveur
import os                               # Lire les variables d'environnement
import time                             # Mesurer la durée des prédictions
from contextlib import asynccontextmanager  # Pour le "lifespan" (démarrage/arrêt)
from pathlib import Path                # Chemins portables

import torch                            # Exécution du modèle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field   # Schémas de validation des entrées/sorties
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

# Racine du projet : ce fichier est dans api/, donc 2 niveaux au-dessus.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# D'OÙ VIENT LE MODÈLE ? os.getenv lit une variable d'environnement :
#   - En LOCAL : la variable MODEL_ID n'existe pas -> on prend la valeur par
#     défaut = le dossier model/sentiment_model/ rempli par train.py.
#   - Sur HUGGING FACE SPACES : on règle MODEL_ID dans les Settings du Space
#     (ex: "rima/avissense-distilcamembert") -> transformers reconnaît que
#     c'est un repo du Hub et TÉLÉCHARGE les poids au démarrage.
# Un seul code, deux environnements : c'est le pattern "12-factor app".
MODEL_ID = os.getenv("MODEL_ID", str(PROJECT_ROOT / "model" / "sentiment_model"))

# Doit correspondre à la longueur utilisée à l'entraînement (train.py) :
# le modèle a appris sur des avis de 256 tokens max, on prédit pareil.
MAX_LENGTH = 256

# Garde-fou : le modèle tronque à 256 tokens de toute façon, mais on rejette
# les entrées absurdement longues AVANT de gaspiller du calcul dessus.
MAX_INPUT_CHARS = 5000

# Logger : chaque événement important (démarrage, prédiction) est tracé dans
# la console du serveur avec l'heure. Indispensable pour diagnostiquer en prod.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("avissense")

# Conteneur global pour le modèle et le tokenizer. Rempli au démarrage par le
# lifespan, lu par les endpoints. (Un simple dict suffit : FastAPI traite les
# requêtes dans un seul processus, pas besoin de mécanisme plus complexe.)
ml_resources = {}


# ─────────────────────────────────────────────────────────────────────────────
# CYCLE DE VIE — chargement du modèle au démarrage du serveur
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Tout ce qui est AVANT le `yield` s'exécute au DÉMARRAGE du serveur ;
    tout ce qui est APRÈS s'exécute à l'ARRÊT.

    C'est le mécanisme officiel de FastAPI pour préparer des ressources
    lourdes (modèle ML, connexion base de données...) une seule fois.
    """
    logger.info("Chargement du modèle : %s ...", MODEL_ID)
    start_time = time.perf_counter()   # Chrono haute précision

    # Mêmes objets que dans scripts/predict.py : le tokenizer (texte->nombres)
    # et le modèle (nombres->logits). from_pretrained accepte indifféremment
    # un dossier local OU un nom de repo du Hub (cf. MODEL_ID plus haut).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()   # Mode évaluation : dropout désactivé -> prédictions stables

    ml_resources["tokenizer"] = tokenizer
    ml_resources["model"] = model
    logger.info("Modèle chargé en %.1f s — API prête.", time.perf_counter() - start_time)

    yield  # ───── l'API tourne ici, entre le démarrage et l'arrêt ─────

    ml_resources.clear()   # Libère la mémoire à l'arrêt du serveur
    logger.info("Ressources libérées, arrêt de l'API.")


# Création de l'application. Les métadonnées (title, description...)
# apparaissent en haut de la documentation Swagger générée sur /docs.
app = FastAPI(
    title="AvisSense — Analyse de sentiment d'avis cinéma en français",
    description="DistilCamemBERT fine-tuné sur le dataset Allociné. "
                "Classification binaire positif/négatif avec score de confiance.",
    version="1.0.0",
    lifespan=lifespan,   # Branche la fonction de cycle de vie ci-dessus
)

# ─── CORS (Cross-Origin Resource Sharing) ───────────────────────────────────
# Par défaut, un navigateur INTERDIT à une page web d'appeler une API
# hébergée sur un autre domaine (protection contre les sites malveillants).
# Notre frontend sera hébergé ailleurs que l'API -> sans ces en-têtes CORS,
# le navigateur bloquerait tous ses appels fetch() vers /predict.
# allow_origins=["*"] = tout domaine autorisé (OK pour un projet d'école ;
# en production réelle, on listerait uniquement le domaine du frontend).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SCHÉMAS Pydantic — le "contrat" d'entrée/sortie de l'API
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic valide AUTOMATIQUEMENT chaque requête contre ces schémas :
# si un client envoie {"texte": ...} au lieu de {"text": ...}, ou un nombre
# au lieu d'une chaîne, FastAPI renvoie une erreur 422 détaillée tout seul.

class ReviewInput(BaseModel):
    """Corps attendu de la requête POST /predict."""
    text: str = Field(
        ...,   # ... signifie "champ OBLIGATOIRE" (pas de valeur par défaut)
        description="L'avis en français à analyser",
        examples=["Ce film est incroyable"],   # Affiché dans Swagger
    )


class PredictionOutput(BaseModel):
    """Réponse renvoyée par POST /predict.

    Déclarer le schéma de SORTIE sert à : (1) documenter le contrat dans
    Swagger pour le dev frontend, (2) garantir que la réponse a toujours
    exactement cette forme.
    """
    label: str = Field(description='"positif" ou "négatif"')
    confidence: float = Field(description="Probabilité de la classe prédite (0 à 1)")
    probabilities: dict[str, float] = Field(
        description="Probabilité de chaque classe (somme = 1)"
    )
    processing_time_ms: float = Field(description="Durée de l'inférence en millisecondes")


# ─────────────────────────────────────────────────────────────────────────────
# INFÉRENCE — les mêmes 4 étapes que scripts/predict.py
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(text: str) -> dict:
    """Tokenise le texte, exécute le modèle et renvoie label + probabilités.

    Étapes (détaillées dans scripts/predict.py) :
        1. tokenisation   texte -> tenseurs de numéros de tokens
        2. forward pass   tenseurs -> logits (2 scores bruts)
        3. softmax        logits -> probabilités (somme = 1)
        4. argmax         classe la plus probable + sa confiance
    """
    tokenizer = ml_resources["tokenizer"]
    model = ml_resources["model"]

    # Étape 1 : texte -> tenseurs PyTorch, tronqué à 256 tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    # Étape 2 : passage dans le réseau, sans calcul de gradients (inférence
    # pure : les gradients ne servent qu'à l'entraînement)
    with torch.no_grad():
        logits = model(**inputs).logits[0]   # [0] : enlève la dimension batch

    # Étape 3 : scores bruts -> probabilités
    probabilities = torch.softmax(logits, dim=-1)

    # Étape 4 : classe gagnante + nom lisible (id2label écrit par train.py)
    predicted_class_id = int(torch.argmax(probabilities))
    label = model.config.id2label[predicted_class_id]   # "positif" / "négatif"

    return {
        "label": label,
        "confidence": round(float(probabilities[predicted_class_id]), 4),
        # On expose AUSSI les deux probabilités : utile au frontend pour
        # afficher une jauge, et plus transparent qu'un score unique.
        "probabilities": {
            model.config.id2label[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["info"])
def api_info():
    """Point d'entrée : décrit l'API.

    Utile au dev frontend et comme test rapide ("est-ce que l'API répond ?").
    Les `tags` regroupent les endpoints par catégorie dans Swagger.
    """
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
    """Vérifie que l'API tourne ET que le modèle est bien en mémoire.

    Usage typique : après un déploiement, on appelle /health pour savoir si
    le Space est prêt (le téléchargement du modèle au boot peut prendre
    ~30 s) ; le frontend peut aussi l'appeler avant d'autoriser l'envoi.
    """
    return {
        "status": "ok",
        "model_loaded": "model" in ml_resources,
        "model_id": MODEL_ID,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict_sentiment(review: ReviewInput):
    """Prédit le sentiment (positif/négatif) d'un avis en français.

    `review: ReviewInput` -> FastAPI parse et valide le JSON entrant tout
    seul. `response_model=PredictionOutput` -> la sortie est validée et
    documentée dans Swagger.

    Erreurs gérées :
        400 — texte vide ou composé uniquement d'espaces
        400 — texte dépassant 5000 caractères
        422 — JSON malformé ou champ "text" manquant (automatique, Pydantic)
        503 — modèle pas encore chargé (Space en cours de démarrage)
    """
    # .strip() enlève les espaces/retours à la ligne au début et à la fin :
    # "   " devient "" et sera bien rejeté comme vide.
    text = review.text.strip()

    # ── Validation de l'entrée ──────────────────────────────────────────────
    # HTTPException interrompt la requête et renvoie le code + le message
    # en JSON : {"detail": "Le texte ne peut pas être vide."}
    if not text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Texte trop long ({len(text)} caractères, max {MAX_INPUT_CHARS}).",
        )
    # Cas rare : requête reçue pendant que le Space charge encore le modèle.
    # 503 = "Service Unavailable" : le client sait qu'il peut réessayer.
    if "model" not in ml_resources:
        raise HTTPException(status_code=503, detail="Modèle en cours de chargement.")

    # ── Prédiction + mesure du temps de traitement ──────────────────────────
    start_time = time.perf_counter()
    result = run_inference(text)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    # Trace serveur : qui demande quoi, résultat, durée. Les 40 premiers
    # caractères suffisent pour identifier la requête sans noyer les logs.
    logger.info('Prédiction : "%s..." -> %s (%.0f %%) en %.0f ms',
                text[:40], result["label"], result["confidence"] * 100, elapsed_ms)

    # ** déplie le dict result dans les champs du modèle Pydantic
    return PredictionOutput(**result, processing_time_ms=elapsed_ms)
