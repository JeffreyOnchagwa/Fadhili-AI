/**
 * Sign languages Fadhili AI supports. KSL is the primary focus and
 * the default language everywhere in the app; ASL and BSL are
 * secondary, additional languages.
 */
export type SupportedSignLanguage = "KSL" | "ASL" | "BSL";

export const SIGN_LANGUAGES: {
  code: SupportedSignLanguage;
  label: string;
  flag: string;
}[] = [
  { code: "KSL", label: "Kenyan Sign Language", flag: "🇰🇪" },
  { code: "ASL", label: "American Sign Language", flag: "🇺🇸" },
  { code: "BSL", label: "British Sign Language", flag: "🇬🇧" },
];

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
  language: SupportedSignLanguage;
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
  language: SupportedSignLanguage;
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
