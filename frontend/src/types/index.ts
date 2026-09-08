/**
 * Fadhili AI is a Kenyan Sign Language platform. KSL is the only
 * language the product supports, and the only one it claims to.
 *
 * An earlier concept advertised ASL and BSL alongside KSL. Nothing was
 * ever trained, verified or licensed for either, so advertising them
 * misrepresented the product. They have been removed rather than left
 * as "coming soon" labels on functionality that does not exist.
 */
export const SIGN_LANGUAGE_CODE = "KSL" as const;
export const SIGN_LANGUAGE_LABEL = "Kenyan Sign Language" as const;

/**
 * Status of a recognition or generation request. "unavailable" is the
 * honest, current state everywhere the ML backend has not been wired
 * up yet — the UI must never fabricate any other status.
 */
export type RecognitionStatus =
  | "unavailable"
  | "idle"
  | "loading"
  | "success"
  | "error";

/**
 * Shape returned by the (future) recognition endpoints. Every field
 * is optional/nullable because, until the backend exists, there is
 * nothing real to display.
 */
export interface RecognitionResult {
  status: RecognitionStatus;
  detectedSign: string | null;
  translation: string | null;
  /** 0–1 confidence score from the model, once one exists. */
  confidence: number | null;
  message: string;
}

export type LessonLevel = "Beginner" | "Intermediate" | "Advanced";

export interface Lesson {
  id: string;
  level: LessonLevel;
  category: string;
  title: string;
  description: string;
  /** Number of items in the lesson (signs, phrases, concepts). */
  itemCount: number;
  /** Whether this lesson is reachable yet in the current build. */
  locked: boolean;
}

export interface SignDictionaryEntry {
  id: string;
  word: string;
  meaning: string;
  videoUrl: string | null;
  difficulty: LessonLevel;
  usageExample: string | null;
  source: string | null;
}

/** Result of a future camera-based practice evaluation. */
export interface PracticeEvaluation {
  status: RecognitionStatus;
  score: number | null;
  feedback: string | null;
}
