# Fadhili AI — Dataset Inventory & Licence Register

Last verified: 2026-09-08.

This document records every data source used by Fadhili AI, what we
verified about it, and what we are and are not permitted to do with it.
Claims here were checked against primary sources (the Kaggle API, the
files on disk) rather than assumed.

---

## 1. Primary dataset — eKitabu KSL Video Dataset

| Field | Value |
|---|---|
| Title | Kenyan Sign Language (KSL) Video Dataset |
| Kaggle slug | `ekitabu/kenyan-sign-language-videos` |
| Dataset ID | 9468606 |
| Uploader | `laurettekazenga` (role: ADMIN) |
| Licence | **CC0-1.0** (Creative Commons Public Domain Dedication) |
| Licence verified via | `kaggle datasets metadata` API response, 2026-09-08 |
| Class directories | 30 |
| Videos (full dataset) | ~2,237 |
| Signers | 15 |
| Resolution | 1920×1080, all files |
| Frame rate | 25 fps, all files |
| Total size | ~14 GB |

### 1.1 Vocabulary split — named vs opaque

Only **15 of the 30** class directories carry a real glossed label. The
remaining 15 are opaque numeric identifiers whose meanings are not
documented anywhere in the dataset.

**Named classes (usable as product vocabulary — 15):**

Agreement, Apple, Colour, Friend, Gift, Market, Monday, Picture, Proud,
Sweater, Teach, Tomatoes, Tortoise, Twin, Ugali

**Opaque classes (NOT usable as product vocabulary — 15):**

No_9, No_17, No_22, No_35, No_48, No_54, No_66, No_73, No_89, No_91,
No_100, No_125, No_268, No_388, No_444

**Policy.** Opaque labels must never be shown to users, mapped to a
guessed gloss, or counted toward advertised vocabulary size. Guessing
what `No_268` means would be fabricating Kenyan Sign Language. They
remain eligible only as auxiliary data for representation learning,
where the gloss is irrelevant to the objective.

### 1.2 Per-class counts

Every named class has 15 signers and 75 videos, except:
`Market` 73 videos, `Teach` 69 videos (14 signers), `No_54` 70 (14 signers).

### 1.3 Critical finding — the dataset is TWO corpora, not one

This was not documented by the uploader and materially affects every
generalization claim made about the data.

| | Domain A | Domain B |
|---|---|---|
| Signers | 01–10 | 11–15 |
| Filename pattern | `Signer_01_1.mov` | `Signer_11_A056.mov` |
| Setting | Controlled studio, uniform backdrop | Real rooms — homes/offices |
| Posture | Standing, full upper body | Seated |
| Lighting | Controlled, high contrast | Natural, variable; several clips over- or under-exposed |
| Mean duration | 4.22 s | 5.24 s |
| Mean file size | 4.47 MB | 12.08 MB |

Consequences:

- A "signer-independent" split that draws train and test from within
  Domain A measures a much easier problem than deployment.
- The 97.6% signer-group CV figure (signers 01–12) and the ~53%
  held-out figure (signers 13–15) are **not measuring the same thing**.
  The gap is substantially a domain gap, not purely a signer gap.
- Any future evaluation must report Domain A and Domain B separately.

### 1.4 MediaPipe detection quality varies enormously by signer

Fraction of hand-landmark values that are zero (detection failure),
measured on the v3 feature cache:

| Signer | 8 | 6 | 14 | 15 | 2 | 7 | 1 | 10 | 3 | 9 | 4 | 12 | 13 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zero fraction | .07 | .10 | .10 | .10 | .12 | .13 | .17 | .22 | .25 | .25 | .35 | .49 | .53 | .62 |

Signer 13 is underexposed with dark clothing; signer 11 is
overexposed. Pose detection is 100% for every signer — only the hands
fail.

Benchmarked remedies (`scripts/bench_extraction.py`): raising
`model_complexity` to 2 and pose-guided upper-body cropping/upscaling
did **not** recover the failed signers (signer 11 stayed at 0%). The
failures are photometric, not resolution-related.

---

## 2. Legal position on the video files

### 2.1 Copyright — clear

CC0-1.0 is a public-domain dedication. It permits commercial use,
redistribution, and modification with no attribution requirement. We
attribute anyway as good practice.

### 2.2 Likeness, privacy and data protection — NOT clear

**This is a genuine open risk and is flagged for professional legal
review before any public release that redistributes the footage.**

CC0 waives the uploader's *copyright*. It does not, and cannot, waive:

- the **personality / likeness rights** of the 15 identifiable people
  filmed;
- obligations under Kenya's **Data Protection Act 2019** — video of an
  identifiable person is personal data, and facial imagery processed to
  identify someone can engage the sensitive-data provisions;
- whether those signers gave informed consent to *redistribution in a
  commercial product*, as opposed to research use.

There is no consent documentation in the dataset. The uploader's CC0
declaration is an assertion we cannot independently verify.

**Operating decision.** We distinguish two uses:

| Use | Status |
|---|---|
| Training models on the footage | Proceeding. Well supported by CC0; the model does not republish the footage. |
| Republishing raw footage as in-app Learn/Dictionary media | **Not proceeding** without legal review. |

For in-app sign demonstrations we instead render **skeletal landmark
animations** derived from the videos. These are derived from CC0 data,
depict no identifiable person, and are labelled honestly as a landmark
rendering of a recorded signer's performance — not as a substitute for
learning from a fluent Deaf signer.

Limitation to state plainly: a skeleton omits facial expression and
fine handshape detail, both of which are grammatically significant in
KSL. The Learn UI must say so rather than implying completeness.

---

## 3. Other sources evaluated

| Source | Status |
|---|---|
| Maseno KSL word-based pose dataset | Lead only. Not obtained, not verified, not used. Must not be cited as a source until licence and contents are confirmed. |
| AI4KSL resources | Lead only. Not obtained, not verified, not used. |

No data was scraped. No copyrighted video was obtained outside the
Kaggle dataset above.

---

## 4. Media, fonts and icons in the application

| Asset | Source | Licence |
|---|---|---|
| `lucide-react` icons | npm, Lucide | ISC |
| Fonts | To be confirmed during the frontend asset audit |  |
| Imagery | No AI-generated imagery is used anywhere in this project. |  |

---

## 5. Model artefacts

| Artefact | Provenance |
|---|---|
| `fadhili_ksl_v3_candidate.keras` | Trained in-project on the CC0 dataset. Retained as the rollback model. |
| `best_model.h5`, `action.h5` (`backend/ksl-model-source/`) | Inherited from the original 1662-feature tutorial pipeline. Superseded; not used by the running application. |
| MediaPipe Holistic | Google, Apache-2.0. Used for landmark extraction at train and inference time. |

No pretrained weights with non-commercial or research-only licences
have been introduced.
