# Fadhili KSL — Model Notes

## Current champion: v5 (BiGRU, 10 classes)

| | |
|---|---|
| Architecture | BiGRU (64+32 units), 184,138 parameters |
| Input | 32 frames × 320 features (150 raw landmark values → normalized, plus first-difference velocity) |
| Classes | Agreement, Friend, Gift, Market, Monday, Picture, Proud, Teach, Twin, Ugali |
| Training signers | 01–10 (studio) |
| Early-stopping signers | 11–12 (in-the-wild, held out from training) |
| Diagnostic signers (evaluated once) | 13–15 (in-the-wild) |
| Diagnostic accuracy | **0.673** (vs v3's 0.533 on the same signers, same classes) |
| Diagnostic macro F1 | 0.699 |
| Diagnostic ECE | 0.199 (vs v3's ~0.40) |
| Worst signer | 0.38 (signer 14) |
| Fitted temperature | 0.894 |

Fallback: `fadhili_ksl_v3_candidate.keras`, preserved untouched, never overwritten.

## What actually fixed the studio → wild collapse

Not architecture. Two measured defects in the input representation
(full account in `README.md` and inline in `backend/app/services/ksl_features.py`):

1. **Fragile hand-scale normalization** — v3 divided every hand
   coordinate by the *projected* wrist-to-middle-MCP distance, which
   collapses under foreshortening. A centroid-based scale nearly halved
   feature-magnitude variance (CV 0.254 → 0.139).
2. **Temporal domain shift** — wild signers sign markedly slower and
   idle far more (signer 14: 26% active frames, less than half the
   studio wrist velocity). Uniform 32-frame resampling squashed the
   real sign into a handful of frames. Trimming to the motion-energy
   interval *before* resampling, plus tempo augmentation (±30% speed),
   was the single largest lever: it took signer 14 from 0.20 to 0.68 on
   an early 5-class ablation.

## A corrected methodological error, disclosed

An earlier version of the experiment harness (`experiments_v5.py`,
`train_fold`) passed the *evaluation* fold to Keras as `validation_data`
with `EarlyStopping(restore_best_weights=True)` — leaking the
evaluation set into model selection. Every affected number was
optimistically biased by roughly 5 points (an initial cross-domain read
of 0.776 corrected to a leak-free 0.724). `train_fold` now carves an
inner signer-grouped validation split out of the *training* signers
only; the evaluation fold is never touched until `evaluate()` runs. The
0.673 figure above is post-correction.

## Rejection: a motion gate, not an 11th class

An explicit "no sign" class was built from the idle footage that motion
trimming discards — real "nobody is signing" data from the same
recordings, at zero extra annotation cost. It underperformed: 46.7%
no-sign recall and roughly 7 points of classification accuracy lost to
the 10-class model.

A pose motion-energy gate at threshold 0.010, calibrated on 248 real
sign windows and 233 real idle windows, does far better: rejects 99.6%
of idle while wrongly rejecting 0.0% of real signs. That is what ships
(`ksl_recognizer_v5.py`, `MOTION_GATE_THRESHOLD`). Concretely: v3
classified pure random noise as "Monday" at 0.71 confidence and
accepted it; v5 returns no prediction for the same input.

Honest limit, stated in the API (`reason` field) and the UI: the gate
detects "not moving," not "moving without signing." A wave or an
unrelated gesture can still be classified as a word.

## Vocabulary expansion — tried, measured, not shipped

The eKitabu dataset has 15 named classes total (10 originally used,
plus Apple, Colour, Sweater, Tomatoes, Tortoise). All 1,105 available
videos across all 15 were extracted and a 15-class champion was
trained and evaluated under the identical protocol above.

| | 10-class champion | 15-class experiment |
|---|---|---|
| Diagnostic accuracy | **0.673** | 0.573 |
| Diagnostic macro F1 | **0.699** | 0.583 |
| Diagnostic ECE | **0.199** | 0.242 |
| Worst per-class F1 | 0.40 (Twin, v3-era) | **0.00 (Proud — total collapse into "Market")** |

The 15-class model is a real regression, not a rounding difference: it
loses "Proud" outright and drags "Market" precision to 0.21 (everything
falsely called Market). Per this project's own standing rule — *"It is
better to have 100 well-supported signs than 1,000 poorly supported
labels"* — that isn't shipped as the recognition model.

**What did ship from the expanded data:** all 15 named signs, including
the 5 new ones, are real, verified KSL vocabulary with genuine recorded
video behind them. They appear in Learn and the Dictionary. The
frontend cross-references the *live* model's vocabulary
(`GET /api/v1/ksl/classes`) against the full 15-word list, so only the
10 the champion actually recognizes get a "recognised on camera" badge
— the other 5 are honestly listed as vocabulary without a false
recognition claim. That is the "ML-recognizable vocabulary" vs.
"educational vocabulary with verified media" split called for by this
project's brief, and it required no product-code change: it was built
that way from the start.

The 15-class experiment's model and metadata are preserved locally
(`backend/training/models/experiments/fadhili_v5_15class_experiment.keras`,
`backend/training/data/metadata/fadhili_v5_15class_experiment.json`)
rather than deleted, in case more per-class data later closes the gap —
Tomatoes, Tortoise and Sweater each had only 69–75 videos across all
signers vs. 75 for every original class, and Tortoise specifically had
zero studio-signer coverage until the final extraction pass. Neither
file is tracked in git (consistent with every other non-champion model
artefact) — reproduce it with the command in `train_champion.py`
omitting `--classes` to include all 15.

## Reproducing this

```bash
cd backend/training/scripts
PY=../../.venv/Scripts/python.exe

# Architecture selection — signer-grouped CV, dev signers 01-12 only.
$PY experiments_v5.py --model bigru --protocol dev_groupcv

# Train and freeze the champion (10 classes; the shipped configuration).
$PY train_champion.py --model bigru --epochs 150 --no-background \
    --classes "Agreement,Friend,Gift,Market,Monday,Picture,Proud,Teach,Twin,Ugali"
```

`SEED = 1337` throughout; the 0.673 figure reproduced exactly across
two independent training runs.
