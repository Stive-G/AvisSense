# Dockerfile — Déploiement sur Hugging Face Spaces (type "Docker")
#
# Sur Spaces, l'application DOIT écouter sur le port 7860.
# Le modèle fine-tuné est téléchargé depuis le Hub au démarrage de l'API,
# via la variable d'environnement MODEL_ID (réglée dans les Settings du Space).

FROM python:3.11-slim

WORKDIR /app

# 1. Installer les dépendances d'abord (couche Docker mise en cache :
#    les rebuilds sont rapides tant que requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copier le code du projet (api/ + frontend/)
COPY . .

# 3. Spaces exécute le conteneur avec un utilisateur non-root :
#    on redirige le cache Hugging Face vers un dossier accessible en écriture
ENV HF_HOME=/tmp/hf_cache

EXPOSE 7860

# Lancement du serveur FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
