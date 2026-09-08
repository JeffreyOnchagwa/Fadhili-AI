import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Camera, Search } from "lucide-react";
import { getLessons } from "../data/lessons";
import type { VocabularyLesson } from "../data/lessons";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { getKSLClasses } from "../services/api";
import type { LessonLevel } from "../types";

const LEVELS: LessonLevel[] = ["Beginner", "Intermediate", "Advanced"];

const LEVEL_STYLES: Record<LessonLevel, string> = {
  Beginner: "bg-teal-100 text-teal-800",
  Intermediate: "bg-ochre-100 text-ochre-800",
  Advanced: "bg-ink/10 text-ink",
};

export default function Learn() {
  const lessons = useMemo(() => getLessons(), []);
  const [activeLevel, setActiveLevel] = useState<LessonLevel>("Beginner");
  const [openLesson, setOpenLesson] = useState<VocabularyLesson | null>(null);
  const [query, setQuery] = useState("");
  const [recognizable, setRecognizable] = useState<Set<string> | null>(null);

  useEffect(() => {
    let cancelled = false;
    void getKSLClasses().then((result) => {
      if (cancelled) return;
      setRecognizable(result.ok ? new Set(result.data.classes) : new Set());
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const totalSigns = lessons.reduce((sum, lesson) => sum + lesson.itemCount, 0);

  const searching = query.trim().length > 0;
  const searchResults = useMemo(() => {
    if (!searching) return [];
    const q = query.trim().toLowerCase();
    return lessons
      .flatMap((lesson) => lesson.entries.map((entry) => ({ lesson, entry })))
      .filter(
        ({ entry }) =>
          entry.gloss.toLowerCase().includes(q) ||
          entry.synonyms.some((s) => s.includes(q))
      );
  }, [lessons, query, searching]);

  const filtered = lessons.filter((lesson) => lesson.level === activeLevel);

  // ---- Lesson detail view -------------------------------------------------

  if (openLesson) {
    return (
      <div className="mx-auto max-w-4xl px-5 py-12">
        <Button
          variant="ghost"
          icon={<ArrowLeft size={16} />}
          onClick={() => setOpenLesson(null)}
        >
          All lessons
        </Button>

        <header className="mt-6">
          <span
            className={`inline-block rounded px-2.5 py-1 text-xs font-semibold ${
              LEVEL_STYLES[openLesson.level]
            }`}
          >
            {openLesson.level}
          </span>
          <h1 className="mt-3 text-3xl font-medium">{openLesson.title}</h1>
          <p className="mt-2 text-ink-soft">{openLesson.description}</p>
          <p className="mt-1 text-sm text-ink-soft">
            {openLesson.itemCount}{" "}
            {openLesson.itemCount === 1 ? "sign" : "signs"} in this lesson.
          </p>
        </header>

        <ul className="mt-8 flex flex-col gap-3">
          {openLesson.entries.map((entry) => {
            const canRecognize = recognizable?.has(entry.id) ?? false;
            return (
              <li key={entry.id}>
                <Card className="flex flex-col gap-2">
                  <div className="flex flex-wrap items-center gap-3">
                    <h2 className="text-lg font-medium">{entry.gloss}</h2>
                    {canRecognize && (
                      <span className="inline-flex items-center gap-1.5 rounded bg-teal-100 px-2 py-1 text-xs font-semibold text-teal-800">
                        <Camera size={12} aria-hidden="true" />
                        Practice on camera
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-ink-soft">{entry.description}</p>
                </Card>
              </li>
            );
          })}
        </ul>

        <p className="mt-8 border-t border-ink/10 pt-5 text-sm text-ink-soft">
          Fadhili holds recorded Kenyan Sign Language video for every sign
          listed here, but does not yet publish those recordings in the app —
          they show identifiable people, and consent for redistribution has not
          been confirmed. Until that is resolved, use this list alongside a
          qualified KSL teacher rather than as a substitute for one.
        </p>
      </div>
    );
  }

  // ---- Catalogue view -----------------------------------------------------

  return (
    <div className="mx-auto max-w-6xl px-5 py-12">
      <header className="max-w-2xl">
        <h1 className="text-3xl font-medium sm:text-4xl">Fadhili Learn</h1>
        <p className="mt-3 text-ink-soft">
          Kenyan Sign Language vocabulary, grouped into lessons. Every sign
          listed is one Fadhili holds real recorded KSL video for — {totalSigns}{" "}
          in total today.
        </p>
      </header>

      <div className="mt-8">
        <label htmlFor="lesson-search" className="sr-only">
          Search signs
        </label>
        <div className="relative max-w-md">
          <Search
            size={16}
            className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-soft"
            aria-hidden="true"
          />
          <input
            id="lesson-search"
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search signs..."
            className="min-h-[44px] w-full rounded-lg border border-ink/15 bg-white pl-10 pr-4 text-base text-ink"
          />
        </div>
      </div>

      {searching ? (
        <section className="mt-8" aria-label="Search results">
          <p className="text-sm text-ink-soft">
            {searchResults.length}{" "}
            {searchResults.length === 1 ? "sign" : "signs"} matching &ldquo;
            {query.trim()}&rdquo;.
          </p>
          <ul className="mt-4 flex flex-col gap-3">
            {searchResults.map(({ lesson, entry }) => (
              <li key={entry.id}>
                <Card className="flex flex-col gap-1.5">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <h2 className="text-lg font-medium">{entry.gloss}</h2>
                    <span className="text-xs text-ink-soft">
                      in {lesson.title}
                    </span>
                  </div>
                  <p className="text-sm text-ink-soft">{entry.description}</p>
                </Card>
              </li>
            ))}
          </ul>
        </section>
      ) : (
        <>
          <div
            className="mt-8 flex gap-2"
            role="tablist"
            aria-label="Lesson level"
          >
            {LEVELS.map((level) => {
              const count = lessons.filter((l) => l.level === level).length;
              return (
                <button
                  key={level}
                  type="button"
                  role="tab"
                  aria-selected={activeLevel === level}
                  onClick={() => setActiveLevel(level)}
                  className={`min-h-[44px] rounded-lg px-4 text-sm font-semibold ${
                    activeLevel === level
                      ? "bg-ink text-paper"
                      : "border border-ink/15 text-ink-soft hover:text-ink"
                  }`}
                >
                  {level}
                  <span className="ml-1.5 font-normal opacity-70">{count}</span>
                </button>
              );
            })}
          </div>

          <div className="mt-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {filtered.map((lesson) => (
              <Card key={lesson.id} className="flex flex-col gap-3">
                <span
                  className={`self-start rounded px-2.5 py-1 text-xs font-semibold ${
                    LEVEL_STYLES[lesson.level]
                  }`}
                >
                  {lesson.category}
                </span>
                <h2 className="text-lg font-medium">{lesson.title}</h2>
                <p className="text-sm text-ink-soft">{lesson.description}</p>
                <p className="text-xs text-ink-soft">
                  {lesson.itemCount}{" "}
                  {lesson.itemCount === 1 ? "sign" : "signs"}
                </p>
                <Button
                  variant="secondary"
                  size="md"
                  className="mt-1 self-start"
                  onClick={() => setOpenLesson(lesson)}
                >
                  Open lesson
                </Button>
              </Card>
            ))}
          </div>
        </>
      )}

      <section className="mt-16" aria-labelledby="practice-heading">
        <Card className="max-w-3xl">
          <h2 id="practice-heading" className="text-2xl font-medium">
            Practising on camera
          </h2>
          <p className="mt-3 text-ink-soft">
            The Interpreter can recognise a subset of this vocabulary from your
            camera, so you can check whether a sign you make is read the way
            you intended.
          </p>
          <p className="mt-3 text-sm text-ink-soft">
            It will not tell you that your signing is correct. The model
            recognises a small vocabulary, was trained mostly on recordings of
            other people in other rooms, and can be confidently wrong. Treat a
            match as a hint, not as assessment.
          </p>
          <Link
            to="/interpreter"
            className="mt-5 inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-ink/15 bg-white px-4 text-sm font-semibold text-ink hover:border-ink/30"
          >
            <Camera size={16} aria-hidden="true" />
            Open the Interpreter
          </Link>
        </Card>
      </section>
    </div>
  );
}
