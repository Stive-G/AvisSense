# =============================================================================
# Dockerfile — Déploiement sur Hugging Face Spaces (type "Docker")
# =============================================================================
#
# QU'EST-CE QU'UN DOCKERFILE ?
#   La recette de construction d'une IMAGE Docker : un environnement complet
#   et figé (OS + Python + dépendances + notre code) qui s'exécute à
#   l'identique partout. Si le conteneur tourne sur ma machine, il tournera
#   exactement pareil sur Spaces — fini le "ça marche chez moi".
#
# POURQUOI DOCKER ICI ?
#   Hugging Face Spaces propose 4 types de SDK : Gradio, Streamlit, statique
#   et Docker. Seul le type Docker permet de lancer un serveur arbitraire —
#   donc notre API FastAPI. Contrainte de Spaces : l'application DOIT
#   écouter sur le port 7860.
#
# OÙ EST LE MODÈLE ?
#   PAS dans l'image (il pèse 270 Mo et changerait à chaque ré-entraînement).
#   Il est sur le HF Hub ; l'API le télécharge au démarrage grâce à la
#   variable d'environnement MODEL_ID (à régler dans Settings > Variables
#   du Space, ex : MODEL_ID=rima/avissense-distilcamembert).
#
# TESTER EN LOCAL (optionnel, nécessite Docker Desktop) :
#   docker build -t avissense .
#   docker run -p 7860:7860 -e MODEL_ID=rima/avissense-distilcamembert avissense
#   -> http://localhost:7860/docs
# =============================================================================

# Image de départ : Python 3.11 officiel, variante "slim" = sans les outils
# superflus -> image finale plus légère et plus rapide à déployer.
FROM python:3.11-slim

# Tous les chemins relatifs des instructions suivantes partent de /app
# (le dossier est créé automatiquement s'il n'existe pas).
WORKDIR /app

# ── ASTUCE DE CACHE DOCKER ───────────────────────────────────────────────────
# Docker construit l'image par COUCHES et met chaque couche en cache : une
# couche n'est reconstruite que si ce qu'elle copie/exécute a changé.
# En copiant requirements.txt SEUL avant le reste du code, l'installation
# des dépendances (l'étape la plus longue, ~5 min : torch est gros) reste en
# cache tant que requirements.txt ne change pas. Modifier api/main.py ne
# redéclenche PAS le pip install -> rebuilds en quelques secondes.
COPY requirements.txt .

# --no-cache-dir : pip ne garde pas les archives téléchargées -> image
# plus petite (le cache pip ne servirait à rien dans une image figée).
RUN pip install --no-cache-dir -r requirements.txt

# Maintenant seulement, copier tout le code du projet (api/, scripts/...).
# Le .gitignore n'est pas utilisé par Docker, mais les poids du modèle ne
# sont de toute façon pas dans le repo (ils sont sur le Hub).
COPY . .

# Spaces exécute le conteneur avec un utilisateur NON-root qui n'a pas le
# droit d'écrire n'importe où. Or transformers télécharge le modèle dans un
# cache (par défaut ~/.cache/huggingface, non accessible en écriture ici).
# On redirige ce cache vers /tmp, accessible en écriture à tout le monde.
ENV HF_HOME=/tmp/hf_cache

# Documente le port utilisé (Spaces s'attend à 7860).
EXPOSE 7860

# La commande lancée au démarrage du conteneur :
#   uvicorn api.main:app  -> le serveur ASGI qui exécute notre app FastAPI
#   --host 0.0.0.0        -> écoute sur TOUTES les interfaces réseau
#                            (avec 127.0.0.1, seul l'intérieur du conteneur
#                             pourrait se connecter : l'API serait invisible)
#   --port 7860           -> le port imposé par Hugging Face Spaces
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
