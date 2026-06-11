# Front React pour Vercel

## Installation

```bash
npm install
```

## Variables d'environnement

Creer un fichier `.env.local` :

```bash
VITE_API_BASE_URL=https://stive-g-avissense.hf.space
```

## Developpement

```bash
npm run dev
```

## Build

```bash
npm run build
```

## Deploiement Vercel

- Importer le dossier `frontend/` comme projet Vercel
- Renseigner `VITE_API_BASE_URL`
- Framework preset : `Vite`
