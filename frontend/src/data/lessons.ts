import type { Lesson, LessonLevel, SupportedSignLanguage } from "../types";

interface LessonBlueprint {
  category: string;
  title: string;
  description: string;
  itemCount: number;
}

const BEGINNER: LessonBlueprint[] = [
  { category: "Alphabet", title: "The alphabet", description: "Fingerspelling every letter, one at a time.", itemCount: 26 },
  { category: "Numbers", title: "Numbers 0–20", description: "Counting and basic quantities.", itemCount: 20 },
  { category: "Greetings", title: "Greetings", description: "Hello, goodbye, and everyday courtesies.", itemCount: 10 },
  { category: "Introductions", title: "Introducing yourself", description: "Names, pronouns, and asking who someone is.", itemCount: 12 },
  { category: "Family", title: "Family", description: "Signs for family members and relationships.", itemCount: 14 },
  { category: "Everyday phrases", title: "Everyday phrases", description: "Short phrases for daily situations.", itemCount: 18 },
];

const INTERMEDIATE: LessonBlueprint[] = [
  { category: "Conversations", title: "Holding a conversation", description: "Turn-taking and follow-up questions.", itemCount: 16 },
  { category: "School", title: "At school", description: "Classroom vocabulary and routines.", itemCount: 15 },
  { category: "Travel", title: "Getting around", description: "Directions, transport, and travel needs.", itemCount: 17 },
  { category: "Healthcare", title: "At the clinic", description: "Describing symptoms and understanding care.", itemCount: 15 },
  { category: "Work", title: "At work", description: "Workplace vocabulary and small talk.", itemCount: 14 },
  { category: "Directions", title: "Directions & places", description: "Asking for and giving directions.", itemCount: 12 },
];

const ADVANCED: LessonBlueprint[] = [
  { category: "Grammar", title: "Sign grammar", description: "Sentence structure specific to signed languages.", itemCount: 12 },
  { category: "Complex sentences", title: "Complex sentences", description: "Combining ideas fluently.", itemCount: 10 },
  { category: "Interpretation", title: "Interpreting basics", description: "Foundations for interpreting between signed and spoken language.", itemCount: 9 },
  { category: "Facial expressions", title: "Facial expressions", description: "How expression changes meaning.", itemCount: 10 },
  { category: "Non-manual markers", title: "Non-manual markers", description: "Grammar carried by the face and body, not the hands.", itemCount: 8 },
  { category: "Conversational fluency", title: "Conversational fluency", description: "Natural pacing in real conversation.", itemCount: 11 },
];

const LEVELS: { level: LessonLevel; blueprints: LessonBlueprint[] }[] = [
  { level: "Beginner", blueprints: BEGINNER },
  { level: "Intermediate", blueprints: INTERMEDIATE },
  { level: "Advanced", blueprints: ADVANCED },
];

/**
 * Builds the lesson catalog for a given sign language. Beginner
 * lessons are unlocked by default; later levels are shown as locked
 * until progress data exists, so the UI never implies content is
 * available that hasn't been built and verified yet.
 */
export function getLessonsForLanguage(language: SupportedSignLanguage): Lesson[] {
  const lessons: Lesson[] = [];
  LEVELS.forEach(({ level, blueprints }) => {
    blueprints.forEach((bp, index) => {
      lessons.push({
        id: `${language}-${level}-${index}`.toLowerCase(),
        language,
        level,
        category: bp.category,
        title: bp.title,
        description: bp.description,
        itemCount: bp.itemCount,
        locked: level !== "Beginner",
      });
    });
  });
  return lessons;
}
