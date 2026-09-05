import type { SignDictionaryEntry } from "../types";

/**
 * The sign-language dictionary. Intentionally empty: Fadhili AI will
 * not display invented KSL/ASL/BSL translations. This array is the
 * exact shape the dictionary UI expects — populate it with verified,
 * attributed entries (ideally reviewed by fluent/Deaf signers) once
 * they exist.
 *
 * Example of the expected shape:
 * {
 *   id: "ksl-thank-you",
 *   word: "Thank you",
 *   meaning: "An expression of gratitude.",
 *   language: "KSL",
 *   videoUrl: "/dictionary/ksl/thank-you.mp4",
 *   difficulty: "Beginner",
 *   usageExample: "Used after receiving help or a gift.",
 *   source: "Reviewed with Kenya National Association of the Deaf",
 * }
 */
export const dictionaryEntries: SignDictionaryEntry[] = [];
