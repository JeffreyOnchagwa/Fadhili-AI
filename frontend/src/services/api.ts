import type {
  PracticeEvaluation,
  RecognitionResult,
  SupportedSignLanguage,
} from "../types";

// Falls back to the local FastAPI dev server if VITE_API_BASE_URL isn't
// set, so requests never resolve to "undefined/...".
const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";

/**
 * True only when VITE_API_BASE_URL was explicitly set. This still
 * gates the older, not-yet-implemented endpoints below (video
 * recognition, text-to-sign, practice evaluation) so they keep
 * showing an honest "not connected" state rather than hitting a
 * FastAPI route that doesn't exist.
 */
export const isBackendConfigured = Boolean(
  import.meta.env.VITE_API_BASE_URL as string | undefined
);

const NOT_CONNECTED_RECOGNITION: RecognitionResult = {
  status: "unavailable",
  detectedSign: null,
  translation: null,
  confidence: null,
  message: "Recognition model not connected",
};

const NOT_CONNECTED_GENERATION_MESSAGE = "Sign generation model not connected";

async function postJson<T>(path: string, body: unknown): Promise<T> {
  if (!API_BASE_URL) {
    throw new Error("VITE_API_BASE_URL is not configured");
  }
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`Request to ${path} failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

/**
 * Sends a full video for sign recognition. Returns an honest
 * "not connected" result until VITE_API_BASE_URL points at a real
 * backend — never a fabricated detection.
 */
export async function recognizeVideo(
  _file: File,
  _language: SupportedSignLanguage
): Promise<RecognitionResult> {
  if (!isBackendConfigured) {
    return NOT_CONNECTED_RECOGNITION;
  }
  try {
    const formData = new FormData();
    formData.append("file", _file);
    formData.append("language", _language);
    const response = await fetch(`${API_BASE_URL}/recognize/video`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) throw new Error("recognizeVideo failed");
    return (await response.json()) as RecognitionResult;
  } catch {
    return {
      status: "error",
      detectedSign: null,
      translation: null,
      confidence: null,
      message: "Couldn't reach the recognition service. Please try again.",
    };
  }
}

/**
 * Sends a single live camera frame for recognition. Intended to be
 * called on an interval once a live model exists.
 */
export async function recognizeFrame(
  _frameDataUrl: string,
  _language: SupportedSignLanguage
): Promise<RecognitionResult> {
  if (!isBackendConfigured) {
    return NOT_CONNECTED_RECOGNITION;
  }
  try {
    return await postJson<RecognitionResult>("/recognize/frame", {
      frame: _frameDataUrl,
      language: _language,
    });
  } catch {
    return {
      status: "error",
      detectedSign: null,
      translation: null,
      confidence: null,
      message: "Couldn't reach the recognition service. Please try again.",
    };
  }
}

export interface SignGenerationResult {
  status: "unavailable" | "success" | "error";
  message: string;
  /** URL of a generated sign-language clip, once generation exists. */
  videoUrl: string | null;
}

/** Translates text into a sign-language sequence. */
export async function translateTextToSign(
  _text: string,
  _language: SupportedSignLanguage
): Promise<SignGenerationResult> {
  if (!isBackendConfigured) {
    return {
      status: "unavailable",
      message: NOT_CONNECTED_GENERATION_MESSAGE,
      videoUrl: null,
    };
  }
  try {
    return await postJson<SignGenerationResult>("/translate/text-to-sign", {
      text: _text,
      language: _language,
    });
  } catch {
    return {
      status: "error",
      message: "Couldn't reach the translation service. Please try again.",
      videoUrl: null,
    };
  }
}

/** Evaluates a user's attempt at a target sign during AI Practice. */
export async function evaluatePracticeSign(
  _frameDataUrl: string,
  _targetSignId: string,
  _language: SupportedSignLanguage
): Promise<PracticeEvaluation> {
  if (!isBackendConfigured) {
    return {
      status: "unavailable",
      score: null,
      feedback: "Practice evaluation isn't available until the ML backend is connected.",
    };
  }
  try {
    return await postJson<PracticeEvaluation>("/practice/evaluate", {
      frame: _frameDataUrl,
      targetSignId: _targetSignId,
      language: _language,
    });
  } catch {
    return {
      status: "error",
      score: null,
      feedback: "Couldn't reach the practice service. Please try again.",
    };
  }
}

// ---------------------------------------------------------------------------
// Real KSL recognition — talks to the working FastAPI backend.
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

/** Returns the 30 user-facing KSL classes the model recognizes. */
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
 * Sends a 30x1662 MediaPipe feature sequence to the KSL model.
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