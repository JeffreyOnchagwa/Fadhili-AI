# Fadhili AI

**Sign. Speak. Connect.**

Kenyan Sign Language recognition and learning. Fadhili reads KSL from a
webcam, maps written and spoken words onto verified KSL signs, and
provides a vocabulary to learn from.

> **Status: experimental research software.** Fadhili recognises a small
> vocabulary, is measurably imperfect on people and settings unlike its
> training data, and is **not** a substitute for a qualified Kenyan Sign
> Language interpreter. Every accuracy figure below is stated with the
> protocol that produced it.

---

## What it does

| Feature | Status |
|---|---|
| KSL → text, live from camera | Working, experimental, 10-sign vocabulary |
| KSL → speech | Working — browser speech synthesis, opt-in |
| Text → KSL | Vocabulary lookup against verified signs; unmatched words are named, never guessed |
| Speech → text → KSL | Working, via the browser's Web Speech API |
| Learn / Dictionary | Verified vocabulary, grouped into lessons |
| Uploaded-video recognition | Not implemented; the UI says so |
| Continuous signing / full translation | Not implemented. Fadhili classifies isolated signs, not sentences. |

Fadhili supports **Kenyan Sign Language only**. An earlier concept
advertised ASL and BSL; nothing was trained, verified or licensed for
either, so they were removed rather than left as "coming soon" labels.

---

## Model performance

Current champion: **Fadhili KSL v5**, a BiGRU over normalized MediaPipe
landmarks. Fallback: **v3**, retained and never overwritten.

The dataset is **two corpora**, and this dominates every number:

- **Domain A** — signers 01–10, controlled studio, standing
- **Domain B** — signers 11–15, real rooms, seated, natural light

Training on Domain A and testing on Domain B is the deployment-relevant
measurement. Anything drawn from within Domain A measures a much easier
problem.

### Studio → wild, 10 classes

| Metric | v3 | v5 |
|---|---|---|
| Accuracy, signers 13–15 | 0.533 | **0.700** |
| Accuracy, all wild signers 11–15 | — | **0.776** |
| Macro F1 | — | 0.786 |
| Expected calibration error | ~0.40 | **0.102** |
| Confidence when wrong | 0.92 | 0.76 |
| Worst signer | 0.16 (s14) | 0.42 (s14) |

Signers 13–15 are a **held-out diagnostic set, not a pristine test
set** — they were inspected repeatedly during v2/v3/v4 development, so
treating them as unbiased would overstate what we know. Architecture
selection uses signer-grouped CV over signers 01–12 only.

### What actually fixed it

Not architecture. Two measured defects in the input representation:

1. **Hand normalization.** v3 divided all 63 hand values by the
   *projected* wrist-to-middle-MCP distance, which collapses under
   foreshortening. A centroid-based scale cut feature-magnitude
   variance nearly in half (CV 0.254 → 0.139).

2. **Temporal domain shift — the dominant factor.** Wild signers sign
   far slower and idle far more (signer 14: 26% active frames at less
   than half the studio wrist velocity). Uniform resampling squashed
   their signs into a handful of frames. Trimming to the active
   interval *before* resampling, plus tempo augmentation, took signer
   14 from 0.20 to 0.68 on a 5-class ablation.

Full detail: [`docs/MODEL.md`](docs/MODEL.md).

---

## Data & licensing

Trained on the [eKitabu Kenyan Sign Language Video Dataset](https://www.kaggle.com/datasets/ekitabu/kenyan-sign-language-videos)
(**CC0-1.0**, verified via the Kaggle API). 15 signers, 1920×1080, 25 fps.

The dataset has 30 class directories but only **15 carry a real glossed
label**. The other 15 are opaque identifiers (`No_9`, `No_268`, …) whose
meanings are undocumented. They are never shown as vocabulary and never
guessed at.

CC0 waives copyright but **not** the filmed signers' likeness or
data-protection rights, and the dataset carries no consent
documentation. Fadhili therefore trains on the footage but does **not**
republish it in-app, pending legal review.

Details: [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md).

---

## Privacy

- Video **never leaves your device.** MediaPipe runs in the browser; only
  numeric landmark arrays are sent for classification.
- The backend stores **nothing** — no database, no request logging, no
  file writes on any request path.
- **No cookies, no localStorage, no analytics, no trackers.** No cookie
  banner is shown because none is warranted.

Details: [`docs/PRIVACY_AND_SECURITY.md`](docs/PRIVACY_AND_SECURITY.md).

---

## Running locally

Requires Python 3.11 and Node 20+.

```bash
# Backend
cd backend
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

Then open the printed URL. Copy `.env.example` to `.env.local` in
`frontend/` and set `VITE_API_BASE_URL` to the backend URL.

---

## Reproducing the model

```bash
cd backend/training/scripts
PY=../../.venv/Scripts/python.exe

# 1. Cache raw MediaPipe landmarks once (~4 h on 2 CPU cores).
#    Run two shards in parallel.
$PY cache_raw_landmarks.py --worker-id 0 --num-workers 2
$PY cache_raw_landmarks.py --worker-id 1 --num-workers 2

# 2. Select an architecture using signer-grouped CV over signers 01-12.
$PY experiments_v5.py --model bigru --protocol dev_groupcv

# 3. Measure the deployment-relevant number.
$PY experiments_v5.py --model bigru --protocol cross_domain
```

Raw landmarks are cached once so that any normalization can then be
ablated in seconds rather than re-running MediaPipe for hours. That
separation is what made the diagnosis above tractable.

---

## Repository layout

```
backend/
  app/                  FastAPI service (stateless inference)
  training/scripts/     Extraction, feature construction, experiments
  training/models/      Model artefacts (v3 fallback tracked)
frontend/
  src/pages/            Interpreter, Translate, Learn, Dictionary, About
  src/services/         API client, MediaPipe feature extraction
  src/data/             Verified KSL vocabulary
docs/                   Model, data/licence, privacy, deployment
```

---

## Known limitations

- Small vocabulary. Fadhili recognises only the signs it has data for.
- Isolated signs only — no continuous signing, no grammar, no
  sentence-level translation.
- Weakest on signers who sign slowly or in low-contrast clothing.
- Confidence is not a probability. It is shown, but labelled as
  uncalibrated, because it stays high on some wrong answers.
- No facial-expression modelling, which carries real grammar in KSL.

## Licence

Code: see `LICENSE`. Dataset: CC0-1.0, attributed above.
