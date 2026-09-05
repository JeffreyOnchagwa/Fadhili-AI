import type {
  PracticeEvaluation,
  RecognitionResult,
  SupportedSignLanguage,
} from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string | undefined;

/** True once a real backend URL has been configured. */
export const isBackendConfigured = Boolean(API_BASE_URL);

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
