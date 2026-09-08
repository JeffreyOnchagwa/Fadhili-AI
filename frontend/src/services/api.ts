// Falls back to the local FastAPI dev server if VITE_API_BASE_URL isn't
// set, so requests never resolve to "undefined/...".
const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";

// ---------------------------------------------------------------------------
// KSL recognition — talks to the FastAPI backend.
//
// This file previously also exported recognizeVideo, recognizeFrame,
// translateTextToSign and evaluatePracticeSign. All four posted to
// routes the backend does not implement (/recognize/video,
// /recognize/frame, /translate/text-to-sign, /practice/evaluate), and
// nothing in the app imported them. They were additionally gated on
// VITE_API_BASE_URL being unset — which is backwards, since that
// variable IS set in production, so in a real deployment they would
// have produced 404s reported to users as "couldn't reach the service".
//
// They have been removed rather than left as dead code. Text-to-KSL is
// served by the verified vocabulary lookup in services/kslVocabulary,
// which only ever shows signs we actually hold data for.
// ---------------------------------------------------------------------------

/** Outcome of a call to the FastAPI backend. Never throws — callers
 * branch on `ok` instead of using try/catch. */
export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: string };

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_status: string;
  supported_languages: string[];
}

export interface KSLClassesResponse {
  language: string;
  classes: string[];
}

export interface KSLPredictionCandidate {
  label: string;
  confidence: number;
}

export interface KSLPredictionResponse {
  language: string;
  prediction: string | null;
  confidence: number;
  experimental: boolean;
  accepted: boolean;
  top_candidates: KSLPredictionCandidate[];
}

export interface KSLPredictionRequest {
  frames: number[][];
}

function isHealthResponse(value: unknown): value is HealthResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.status === "string" &&
    typeof v.model_loaded === "boolean" &&
    typeof v.model_status === "string" &&
    Array.isArray(v.supported_languages)
  );
}

function isKSLClassesResponse(value: unknown): value is KSLClassesResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.language === "string" && Array.isArray(v.classes);
}

function isKSLPredictionResponse(value: unknown): value is KSLPredictionResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.language === "string" &&
    (v.prediction === null || typeof v.prediction === "string") &&
    typeof v.confidence === "number" &&
    typeof v.experimental === "boolean" &&
    typeof v.accepted === "boolean" &&
    Array.isArray(v.top_candidates)
  );
}

async function extractErrorDetail(response: Response, fallback: string): Promise<string> {
  try {
    const body = (await response.json()) as unknown;
    if (
      typeof body === "object" &&
      body !== null &&
      typeof (body as Record<string, unknown>).detail === "string"
    ) {
      return (body as Record<string, unknown>).detail as string;
    }
  } catch {
    // Response wasn't JSON — fall through to the generic message.
  }
  return fallback;
}

/** Checks backend and KSL model status. */
export async function checkHealth(): Promise<ApiResult<HealthResponse>> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      return {
        ok: false,
        error: await extractErrorDetail(response, `Backend returned status ${response.status}`),
      };
    }
    const data = (await response.json()) as unknown;
    if (!isHealthResponse(data)) {
      return { ok: false, error: "Backend returned an unexpected response shape." };
    }
    return { ok: true, data };
  } catch {
    return { ok: false, error: "Couldn't reach the backend. Is it running?" };
  }
}

/** Returns the KSL vocabulary the current model recognizes. */
export async function getKSLClasses(): Promise<ApiResult<KSLClassesResponse>> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/ksl/classes`);
    if (!response.ok) {
      return {
        ok: false,
        error: await extractErrorDetail(response, `Backend returned status ${response.status}`),
      };
    }
    const data = (await response.json()) as unknown;
    if (!isKSLClassesResponse(data)) {
      return { ok: false, error: "Backend returned an unexpected response shape." };
    }
    return { ok: true, data };
  } catch {
    return { ok: false, error: "Couldn't reach the backend. Is it running?" };
  }
}

/**
 * Sends a MediaPipe feature sequence to the KSL model.
 * The model is experimental — a response is never fabricated, and a
 * network or backend failure is reported as an error rather than
 * silently retried or guessed at.
 */
export async function predictKSL(frames: number[][]): Promise<ApiResult<KSLPredictionResponse>> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/ksl/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames } satisfies KSLPredictionRequest),
    });
    if (!response.ok) {
      return {
        ok: false,
        error: await extractErrorDetail(response, `Backend returned status ${response.status}`),
      };
    }
    const data = (await response.json()) as unknown;
    if (!isKSLPredictionResponse(data)) {
      return { ok: false, error: "Backend returned an unexpected response shape." };
    }
    return { ok: true, data };
  } catch {
    return { ok: false, error: "Couldn't reach the backend. Is it running?" };
  }
}
