# Fadhili AI — Privacy, Cookies & Security Audit

Last audited: 2026-09-08, against the code in this repository.

This is an engineering audit, not legal advice. Points needing
professional review are marked **[LEGAL]**.

---

## 1. What Fadhili actually processes

### 1.1 Camera

The camera stream is opened by the browser and **never leaves the
device as video**. MediaPipe Holistic runs client-side (WASM) and
converts each frame into landmark coordinates. Only a numeric array is
sent to the backend.

| | |
|---|---|
| Video frames transmitted | None |
| Video frames stored | None |
| Sent to the backend | One array of landmark values per prediction |
| Retained by the backend | Nothing — see 2.1 |

A prediction request contains no image, no audio, and nothing that
identifies the person. It is a short sequence of normalized joint
coordinates.

### 1.2 Microphone

Used only on the Translate page, only after the user presses "Start
listening", and only via the browser's own `SpeechRecognition` API.

**[LEGAL]** Note for the privacy policy: in Chrome and Edge this Web
Speech API implementation sends audio to Google's speech servers for
transcription. That is browser behaviour Fadhili cannot control or
intercept, but users must be told it happens, because it means audio
does leave the device even though Fadhili never receives it. Fadhili
itself stores neither the audio nor the transcript.

### 1.3 Uploaded video

The Interpreter's upload mode creates a local object URL for preview
only. Nothing is uploaded to any server.

---

## 2. Data retention

### 2.1 Backend

Verified by reading `backend/app/`:

- No database, no ORM, no file writes on any request path.
- No request body is written to logs. The prediction route logs only
  an exception message on failure.
- The unhandled-exception handler returns a generic
  `"Internal server error."` and never leaks stack traces to clients.
- Landmark arrays exist only in memory for the duration of a request.

### 2.2 Browser storage

Audited by grep across `frontend/src`:

| Mechanism | Used? |
|---|---|
| Cookies (`document.cookie`) | **No** |
| `localStorage` | **No** |
| `sessionStorage` | **No** |
| IndexedDB | **No** |

The one previous `localStorage` key (`fadhili:sign-language`, storing
the chosen sign language) was removed along with the ASL/BSL language
switcher, since Fadhili now supports only KSL.

---

## 3. Cookie banner: not required

Fadhili sets **no cookies and no equivalent client-side storage**, and
runs no analytics, advertising or tracking scripts.

Consent requirements under the EU ePrivacy Directive and Kenya's Data
Protection Act 2019 attach to storing or accessing information on a
user's device, and to processing personal data. Fadhili does neither
for tracking purposes.

**A cookie banner would therefore be theatre, and has deliberately not
been added.** This conclusion must be revisited the moment any of the
following is introduced: analytics, embedded third-party media,
accounts or sessions, A/B testing, or a CDN that sets cookies.

Camera and microphone access are gated by the browser's own permission
prompt, which is the appropriate consent mechanism for device access,
and neither is requested until the user activates the feature.

---

## 4. Third parties

| Third party | Purpose | Data exposed | Notes |
|---|---|---|---|
| Google Fonts (`fonts.googleapis.com`, `fonts.gstatic.com`) | Fraunces, Manrope | Visitor IP, User-Agent | **[LEGAL]** German courts have found embedding Google Fonts without consent unlawful under GDPR. Self-hosting the two families removes this exposure entirely and is recommended before an EU-facing launch. |
| MediaPipe Holistic | Landmark extraction | None — runs locally | Loaded as a static asset; performs no network calls of its own once loaded. |

No analytics, no advertising, no session replay, no fingerprinting, no
social embeds, no tag manager. This was verified by grep, not assumed.

---

## 5. Security review

### Verified good

- **No secrets in the repository.** `git grep` for key/secret/token
  patterns returns only false positives (`page_token`, "design tokens").
  Only `.env.example` files are tracked; real `.env` files are ignored.
- **Input validation is strict.** `PredictRequest` enforces exact
  sequence and feature lengths and explicitly rejects NaN and Infinity,
  which Python's JSON parser would otherwise accept and which would
  silently corrupt model input.
- **Request size limit.** Bodies over 2 MB are rejected before parsing.
- **No stack traces leak** to clients.
- **No SQL, no shell execution, no filesystem paths derived from user
  input** anywhere on a request path, so injection and path traversal
  have no surface.
- **No `dangerouslySetInnerHTML`** in the frontend, so no injected-HTML
  XSS vector.
- **CORS is restricted** to configured origins, with credentials
  disabled and methods/headers narrowed, since the API has no auth or
  cookies.

### Open items before public launch

1. **`CORS_ORIGINS` must be set in production.** It defaults to
   `localhost:5174`. If the deployed frontend origin is not added, every
   browser request fails.
2. **No rate limiting.** Inference is CPU-bound (~120 ms per request on
   the current hardware), so an unauthenticated flood is a cheap
   denial-of-service. A per-IP limit belongs at the edge (the hosting
   platform's) or in middleware before public launch.
3. **Self-host the fonts** — see the Google Fonts row above.
4. **HTTPS must be enforced** by the hosting platform. Browsers refuse
   camera and microphone access on non-secure origins anyway, so an
   HTTP deployment would break the core feature.

---

## 6. Dataset privacy

The training videos show 15 identifiable people. The dataset is CC0-1.0,
which waives copyright but not the filmed people's likeness or
data-protection rights, and the dataset carries no consent
documentation.

**[LEGAL]** Training on this footage is well supported. Republishing it
inside the product is not, and is therefore not done — see
`DATA_AND_LICENSES.md` §2.2.

No user recording has ever been used for training, and there is no
mechanism by which one could be: the backend does not retain
submissions.
