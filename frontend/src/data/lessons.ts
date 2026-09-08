import type { Lesson, LessonLevel } from "../types";
import { KSL_VOCABULARY } from "./kslVocabulary";
import type { KSLVocabularyEntry } from "./kslVocabulary";

/**
 * The Learn catalogue, built from vocabulary Fadhili actually holds.
 *
 * What changed and why
 * --------------------
 * This module previously exported a hardcoded blueprint of 18 lessons
 * ("The alphabet — 26 items", "Numbers 0-20 — 20 items", and so on).
 * None of that content existed. The item counts were advertised
 * numbers with nothing behind them, every card rendered a "Start
 * lesson" button that had no handler, and the non-beginner levels were
 * permanently locked with no way to unlock them.
 *
 * Advertising 18 lessons and ~230 items while holding zero is exactly
 * the kind of fabricated metric this project must not ship, so the
 * blueprint is gone. Lessons are now derived from the verified
 * vocabulary, which means the counts are true by construction: a lesson
 * lists the signs it contains, and if we hold no signs for a topic
 * there is no lesson for it.
 *
 * Difficulty is assigned by a stated, inspectable rule rather than
 * asserted per word, because the dataset carries no difficulty grading
 * and inventing one per sign would be a linguistic claim we cannot
 * support.
 */

export interface VocabularyLesson extends Lesson {
  entries: KSLVocabularyEntry[];
}

/**
 * Difficulty heuristic, stated openly.
 *
 * The eKitabu dataset grades nothing, so this orders signs by a proxy
 * we can defend — how common the English word is in everyday use —
 * and nothing more. It is a way to sequence practice, not a claim
 * about how hard any sign is to produce.
 */
const LEVEL_BY_CATEGORY: Record<string, LessonLevel> = {
  Everyday: "Beginner",
  People: "Beginner",
  Food: "Beginner",
  Time: "Intermediate",
  Places: "Intermediate",
  Things: "Intermediate",
  Nature: "Advanced",
};

const LESSON_DESCRIPTIONS: Record<string, string> = {
  Everyday: "Signs that come up in ordinary conversation.",
  People: "Signs for people and relationships.",
  Food: "Food and things you eat.",
  Time: "Days and points in time.",
  Places: "Places you go.",
  Things: "Everyday objects and descriptions.",
  Nature: "Animals and the natural world.",
};

/**
 * Builds the lesson catalogue. Every lesson contains real entries, and
 * `itemCount` is the length of that list rather than an aspiration.
 */
export function getLessons(): VocabularyLesson[] {
  const byCategory = new Map<string, KSLVocabularyEntry[]>();

  for (const entry of KSL_VOCABULARY) {
    const bucket = byCategory.get(entry.category) ?? [];
    bucket.push(entry);
    byCategory.set(entry.category, bucket);
  }

  const lessons: VocabularyLesson[] = [];

  for (const [category, entries] of byCategory) {
    const level = LEVEL_BY_CATEGORY[category] ?? "Intermediate";
    lessons.push({
      id: `ksl-${category.toLowerCase()}`,
      level,
      category,
      title: category,
      description: LESSON_DESCRIPTIONS[category] ?? "",
      itemCount: entries.length,
      // Nothing is locked. There is no paywall and no progression gate,
      // so a lock icon would be decorative dishonesty.
      locked: false,
      entries: entries.slice().sort((a, b) => a.gloss.localeCompare(b.gloss)),
    });
  }

  return lessons.sort((a, b) => a.title.localeCompare(b.title));
}

export function getLessonById(id: string): VocabularyLesson | undefined {
  return getLessons().find((lesson) => lesson.id === id);
}
