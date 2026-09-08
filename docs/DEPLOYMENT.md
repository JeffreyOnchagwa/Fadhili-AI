# Fadhili AI — Deployment

## Architecture

| Piece | What it does | Where it goes |
|---|---|---|
| Frontend (Vite/React SPA) | UI, camera, MediaPipe landmark extraction | Vercel (static) |
| Backend (FastAPI + Keras) | Sign classification from landmark arrays | A container host (see below) |

MediaPipe runs **in the browser**, so no video ever reaches the server.
The API receives only numeric landmark arrays. That keeps the server
small, stateless and cheap.

---

## 1. Frontend — Vercel

`frontend/vercel.json` is committed and configures SPA rewrites,
long-lived caching for hashed assets, and security headers
(`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`,
`Permissions-Policy`, HSTS).

`Permissions-Policy` allows `camera=(self)` and `microphone=(self)` —
both are required for the Interpreter and Translate pages and would
silently break if tightened further.

**Project settings:** root directory `frontend`.

**Environment variables:**

| Variable | Value | Why |
|---|---|---|
| `VITE_API_BASE_URL` | Public URL of the deployed backend | Without it the app calls `127.0.0.1:8000` and every prediction fails. |
| `VITE_SITE_URL` | Public site URL, no trailing slash | Fills the canonical and Open Graph URLs, and enables `sitemap.xml` generation at build time. |

---

## 2. Backend — container

`backend/Dockerfile` builds a CPU-only image from
`requirements-serve.txt`, which deliberately **omits mediapipe and
opencv** (~200 MB saved) since extraction happens client-side. It runs
as a non-root user and honours an injected `$PORT`.

The image is still ~1.2 GB because TensorFlow is large.

### Host options, honestly assessed

| Host | Free tier | Fit |
|---|---|---|
| **Hugging Face Spaces (Docker)** | Yes — 2 vCPU, 16 GB RAM | **Best free fit.** Comfortably fits TensorFlow, purpose-built for ML demos. Free Spaces sleep when idle and cold-start slowly. |
| Render | Free web service, 512 MB RAM | Tight for TensorFlow; risks OOM. Sleeps on idle. |
| Railway | Trial credit, then paid | Works, but is not free on an ongoing basis. |
| Fly.io | Paid | Works; no meaningful free allowance now. |

**Required environment variable:**

| Variable | Value |
|---|---|
| `CORS_ORIGINS` | The exact frontend origin, e.g. `https://fadhili.vercel.app` |

`CORS_ORIGINS` defaults to `localhost:5174`. **If it is not set to the
deployed frontend origin, every browser request fails CORS.** This is
the single most likely deployment mistake.

### A better long-term shape

The model is tiny (~1.3 MB, ~107k parameters). Converting it to
TensorFlow.js and running inference in the browser would remove the
backend entirely — no hosting cost, no cold starts, no network latency,
and nothing leaving the device at all. That is the recommended next
step, and is recorded as future work rather than done, because
promoting a validated model took priority.

---

## 3. Custom domain — the honest position

**There is no longer any legitimate free way to own a top-level custom
domain.** Anything advertised as one is either a subdomain of someone
else's domain or not trustworthy.

- **Freenom**, which was the free `.tk`/`.cf`/`.ga`/`.ml` provider,
  stopped free registrations in 2024 after Meta sued it (its domains
  accounted for ~14% of global phishing in 2023) and ICANN terminated
  its accreditation. It returned in 2026 as a **paid** registrar from
  about €8.22/year. It is not a free option and its reputation is poor.

What is genuinely free are **subdomains** someone else owns and
delegates:

| Option | Example | Notes |
|---|---|---|
| **is-a.dev** | `fadhili.is-a.dev` | Free, legitimate, requested by pull request on GitHub. Best free fit for this project. |
| **eu.org** | `fadhili.eu.org` | Free and long-established, aimed at individuals and non-profits. Approval is manual and can take a while. |
| **js.org** | — | **Fadhili would not qualify.** js.org accepts JavaScript libraries and tooling, not applications. |
| Vercel | `fadhili.vercel.app` | Free and automatic, but a hosting subdomain, not domain ownership. |

To actually **own** the name, it must be bought. For a Kenyan product
`fadhili.co.ke` is the natural choice; `.ai` is available but expensive.
Both cost money, so this is your decision — nothing here has been
purchased.

**Recommendation:** launch on `fadhili.vercel.app`, and if a nicer free
name is wanted, request `fadhili.is-a.dev`. Buy `fadhili.co.ke` when
the project is ready to be presented as a product.

---

## 4. Pre-launch checklist

- [ ] `CORS_ORIGINS` set on the backend to the real frontend origin
- [ ] `VITE_API_BASE_URL` and `VITE_SITE_URL` set on Vercel
- [ ] `/health` returns `model_loaded: true` on the deployed backend
- [ ] Camera works on the deployed HTTPS origin
- [ ] Rate limiting added (see `PRIVACY_AND_SECURITY.md` §5)
- [ ] Google Fonts self-hosted, or accepted as a documented risk
- [ ] Legal review of the dataset likeness question if any recorded
      footage is ever published in-app

Sources for the domain findings:
- https://domainincite.com/31821-freenom-is-back-but-no-longer-free
- https://is-a.dev/
- https://nic.eu.org/
