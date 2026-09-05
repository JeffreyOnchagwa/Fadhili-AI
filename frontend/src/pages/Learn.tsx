import { useMemo, useState } from "react";
import { Lock, Camera, ArrowRight } from "lucide-react";
import { useSignLanguage } from "../context/LanguageContext";
import { getLessonsForLanguage } from "../data/lessons";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { SIGN_LANGUAGES } from "../types";
import type { LessonLevel } from "../types";

const LEVELS: LessonLevel[] = ["Beginner", "Intermediate", "Advanced"];

const LEVEL_STYLES: Record<LessonLevel, string> = {
  Beginner: "bg-teal-100 text-teal-700",
  Intermediate: "bg-ochre-100 text-ochre-700",
  Advanced: "bg-ink/10 text-ink",
};

export default function Learn() {
  const { language, setLanguage } = useSignLanguage();
  const [activeLevel, setActiveLevel] = useState<LessonLevel>("Beginner");
  const lessons = useMemo(() => getLessonsForLanguage(language), [language]);
  const filtered = lessons.filter((lesson) => lesson.level === activeLevel);

  return (
    <div className="mx-auto max-w-6xl px-5 py-12">
      <header className="flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
        <div className="max-w-xl">
          <h1 className="text-3xl font-medium sm:text-4xl">Fadhili Learn</h1>
          <p className="mt-3 text-ink-soft">
            Learn sign language one conversation at a time.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label htmlFor="learn-language" className="text-sm font-semibold text-ink-soft">
            Language
          </label>
          <select
            id="learn-language"
            value={language}
            onChange={(event) => setLanguage(event.target.value as typeof language)}
            className="min-h-[44px] rounded-full border border-ink/15 bg-white px-4 text-sm font-semibold"
          >
            {SIGN_LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.flag} {lang.code}
              </option>
            ))}
          </select>
        </div>
      </header>

      <div className="mt-8 flex gap-2" role="tablist" aria-label="Lesson level">
        {LEVELS.map((level) => (
          <button
            key={level}
            role="tab"
            aria-selected={activeLevel === level}
            onClick={() => setActiveLevel(level)}
            className={`min-h-[44px] rounded-full px-4 text-sm font-semibold ${
              activeLevel === level ? "bg-ink text-paper" : "border border-ink/15 text-ink-soft"
            }`}
          >
            {level}
          </button>
        ))}
      </div>

      <div className="mt-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((lesson) => (
          <Card
            key={lesson.id}
            className={`flex flex-col gap-3 ${lesson.locked ? "opacity-70" : "hover:shadow-soft transition-shadow"}`}
          >
            <div className="flex items-center justify-between">
              <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${LEVEL_STYLES[lesson.level]}`}>
                {lesson.category}
              </span>
              {lesson.locked && <Lock size={16} className="text-ink-soft" aria-hidden="true" />}
            </div>
            <h3 className="text-lg font-medium">{lesson.title}</h3>
            <p className="text-sm text-ink-soft">{lesson.description}</p>
            <p className="text-xs text-ink-soft">{lesson.itemCount} items</p>
            <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-ink/10">
              <div className="h-full w-0 rounded-full bg-teal-500" />
            </div>
            <Button
              variant={lesson.locked ? "secondary" : "ghost"}
              size="md"
              disabled={lesson.locked}
              className="mt-1 self-start"
            >
              {lesson.locked ? "Locked" : "Start lesson"}
            </Button>
          </Card>
        ))}
      </div>

      <section className="mt-16" aria-labelledby="ai-practice-heading">
        <Card className="grid gap-8 lg:grid-cols-[1fr_1.1fr] lg:items-center">
          <div>
            <h2 id="ai-practice-heading" className="text-2xl font-medium">
              AI Practice
            </h2>
            <p className="mt-3 text-ink-soft">
              See a target sign, try it in front of your camera, and get
              feedback on how close you were. Automated feedback needs the ML
              backend, which isn't connected in this build yet.
            </p>
            <ol className="mt-5 flex flex-col gap-2 text-sm text-ink-soft">
              <li className="flex items-center gap-2">
                <ArrowRight size={14} className="text-teal-600" aria-hidden="true" /> See a target sign
              </li>
              <li className="flex items-center gap-2">
                <ArrowRight size={14} className="text-teal-600" aria-hidden="true" /> Sign it back on camera
              </li>
              <li className="flex items-center gap-2">
                <ArrowRight size={14} className="text-teal-600" aria-hidden="true" /> Get feedback (coming soon)
              </li>
            </ol>
          </div>
          <EmptyState
            icon={<Camera size={28} />}
            title="Practice evaluation not connected"
            description="The camera interface is ready — automated scoring will appear here once the practice model is connected."
          />
        </Card>
      </section>
    </div>
  );
}
