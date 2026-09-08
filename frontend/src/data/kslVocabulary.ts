/**
 * The verified Kenyan Sign Language vocabulary Fadhili AI holds data
 * for.
 *
 * Rules this file exists to enforce
 * --------------------------------
 * 1. Every entry corresponds to real recorded KSL video in the eKitabu
 *    KSL dataset (Kaggle `ekitabu/kenyan-sign-language-videos`, CC0-1.0).
 *    Nothing here is invented, and no sign is added to make a number
 *    look bigger.
 *
 * 2. `description` describes the ENGLISH word only. It is deliberately
 *    not a claim about KSL grammar, handshape, regional variation or
 *    usage — none of that is documented in the dataset, and guessing it
 *    would be fabricating a language.
 *
 * 3. The dataset also contains 15 classes labelled only as `No_9`,
 *    `No_17`, `No_268` and so on. Their meanings are undocumented, so
 *    they are deliberately absent from this file. They may be used as
 *    unlabelled data for representation learning, never as vocabulary.
 *
 * 4. `category` is Fadhili's own UI grouping to make browsing possible.
 *    It is an organisational convenience, not a linguistic taxonomy.
 *
 * Which of these the live model can actually recognise is NOT hardcoded
 * here — it is read from the backend at runtime via `getKSLClasses()`,
 * so the UI can never claim recognition support the deployed model
 * does not have.
 */

export type VocabularyCategory =
  | "Everyday"
  | "People"
  | "Food"
  | "Time"
  | "Places"
  | "Things"
  | "Nature";

export interface KSLVocabularyEntry {
  /** Matches the dataset class name and the model's output label. */
  id: string;
  /** Word as shown to the user. */
  gloss: string;
  /** Meaning of the English word. Not a claim about KSL. */
  description: string;
  category: VocabularyCategory;
  /** Alternate spellings/forms matched when translating typed text. */
  synonyms: string[];
}

export const KSL_VOCABULARY: KSLVocabularyEntry[] = [
  {
    id: "Agreement",
    gloss: "Agreement",
    description: "Shared acceptance of a decision or arrangement.",
    category: "Everyday",
    synonyms: ["agree", "agreed", "agreeing"],
  },
  {
    id: "Apple",
    gloss: "Apple",
    description: "The fruit.",
    category: "Food",
    synonyms: ["apples"],
  },
  {
    id: "Colour",
    gloss: "Colour",
    description: "Colour in general, rather than any specific one.",
    category: "Things",
    synonyms: ["color", "colours", "colors"],
  },
  {
    id: "Friend",
    gloss: "Friend",
    description: "A person you know well and like.",
    category: "People",
    synonyms: ["friends", "friendship"],
  },
  {
    id: "Gift",
    gloss: "Gift",
    description: "Something given to someone without expecting payment.",
    category: "Things",
    synonyms: ["gifts", "present"],
  },
  {
    id: "Market",
    gloss: "Market",
    description: "A place where goods are bought and sold.",
    category: "Places",
    synonyms: ["markets", "soko"],
  },
  {
    id: "Monday",
    gloss: "Monday",
    description: "The day of the week following Sunday.",
    category: "Time",
    synonyms: ["mondays"],
  },
  {
    id: "Picture",
    gloss: "Picture",
    description: "A painting, drawing or photograph.",
    category: "Things",
    synonyms: ["pictures", "photo", "photograph", "image"],
  },
  {
    id: "Proud",
    gloss: "Proud",
    description: "Feeling satisfaction in an achievement or person.",
    category: "Everyday",
    synonyms: ["pride"],
  },
  {
    id: "Sweater",
    gloss: "Sweater",
    description: "A knitted garment worn on the upper body.",
    category: "Things",
    synonyms: ["sweaters", "jumper", "pullover"],
  },
  {
    id: "Teach",
    gloss: "Teach",
    description: "To help someone learn something.",
    category: "Everyday",
    synonyms: ["teaches", "teaching", "taught", "teacher"],
  },
  {
    id: "Tomatoes",
    gloss: "Tomatoes",
    description: "The fruit used widely as a vegetable in cooking.",
    category: "Food",
    synonyms: ["tomato"],
  },
  {
    id: "Tortoise",
    gloss: "Tortoise",
    description: "A slow-moving land reptile with a shell.",
    category: "Nature",
    synonyms: ["tortoises", "turtle"],
  },
  {
    id: "Twin",
    gloss: "Twin",
    description: "One of two children born at the same birth.",
    category: "People",
    synonyms: ["twins"],
  },
  {
    id: "Ugali",
    gloss: "Ugali",
    description:
      "A stiff maize-meal staple eaten widely in Kenya and the region.",
    category: "Food",
    synonyms: ["posho"],
  },
];

/** Lookup from any accepted written form to its vocabulary entry. */
const LOOKUP: Map<string, KSLVocabularyEntry> = (() => {
  const map = new Map<string, KSLVocabularyEntry>();
  for (const entry of KSL_VOCABULARY) {
    map.set(entry.gloss.toLowerCase(), entry);
    for (const synonym of entry.synonyms) {
      map.set(synonym.toLowerCase(), entry);
    }
  }
  return map;
})();

export interface TranslatedToken {
  /** The word exactly as the user wrote it. */
  original: string;
  /** The verified sign, or null when we hold no sign for this word. */
  entry: KSLVocabularyEntry | null;
}

/**
 * Maps written text onto verified KSL signs.
 *
 * This is deliberately a vocabulary lookup and NOT a translation
 * engine. It does not reorder words into KSL grammar, handle
 * classifiers, inflection or non-manual markers, and it must not be
 * presented to users as producing a grammatical KSL sentence. Words we
 * hold no verified sign for come back with `entry: null` so the UI can
 * say so plainly instead of substituting something plausible.
 */
export function translateTextToKSL(text: string): TranslatedToken[] {
  const words = text
    .split(/\s+/)
    .map((word) => word.trim())
    .filter(Boolean);

  return words.map((word) => {
    const key = word.toLowerCase().replace(/[^a-z]/g, "");
    return { original: word, entry: LOOKUP.get(key) ?? null };
  });
}
