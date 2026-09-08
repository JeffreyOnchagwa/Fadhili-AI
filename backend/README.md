---
title: Fadhili AI Backend
emoji: 🤟
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Fadhili AI — backend

Stateless FastAPI service that classifies Kenyan Sign Language from a
sequence of MediaPipe landmarks. Landmark extraction runs entirely in
the caller's browser; this API never receives video or images, only
numeric arrays, and stores nothing between requests.

Full project context, model methodology, licensing and privacy notes
live in the main repository:
https://github.com/JeffreyOnchagwa/Fadhili-AI

## Endpoints

- `GET /health` — status and which model is loaded (`v5` champion or
  `v3` fallback)
- `GET /api/v1/ksl/classes` — vocabulary the live model recognises
- `POST /api/v2/ksl/predict` — raw-landmark prediction (current model)
- `GET /docs` — interactive API docs (Swagger UI)

## Required environment variable

`CORS_ORIGINS` must be set to the deployed frontend's exact origin
(e.g. `https://fadhili-ai.vercel.app`) or browser requests from that
origin will be rejected. Set it under this Space's **Settings → Variables
and secrets**.
