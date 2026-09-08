import { useEffect, useMemo, useState } from "react";
import { Search, Camera } from "lucide-react";
import { KSL_VOCABULARY } from "../data/kslVocabulary";
import type { VocabularyCategory } from "../data/kslVocabulary";
import { Card } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";
import { getKSLClasses } from "../services/api";

const CATEGORIES: VocabularyCategory[] = [
  "Everyday",
  "People",
  "Food",
  "Time",
  "Places",
  "Things",
  "Nature",
];

export default function Dictionary() {
  const [query, setQuery] = useState("");
  const [activeCategories, setActiveCategories] = useState<
    VocabularyCategory[]
  >([]);
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

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    return KSL_VOCABULARY.filter((entry) => {
      const matchesQuery =
        q.length === 0 ||
        entry.gloss.toLowerCase().includes(q) ||
        entry.synonyms.some((synonym) => synonym.includes(q));
      const matchesCategory =
        activeCategories.length === 0 ||
        activeCategories.includes(entry.category);
      return matchesQuery && matchesCategory;
    });
  }, [query, activeCategories]);

  const toggleCategory = (category: VocabularyCategory) => {
    setActiveCategories((prev) =>
      prev.includes(category)
        ? prev.filter((c) => c !== category)
        : [...prev, category]
    );
  };

  return (
    <div className="mx-auto max-w-6xl px-5 py-12">
      <header className="max-w-2xl">
        <h1 className="text-3xl font-medium sm:text-4xl">Dictionary</h1>
        <p className="mt-3 text-ink-soft">
          Every Kenyan Sign Language sign Fadhili holds verified recordings
          for. {KSL_VOCABULARY.length} entries today — nothing here is guessed
          or generated, and the list grows only when real data does.
        </p>
      </header>

      <div className="mt-8 flex flex-col gap-4">
        <div className="relative max-w-md">
          <Search
            size={18}
            className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-ink-soft"
            aria-hidden="true"
          />
          <label htmlFor="dictionary-search" className="sr-only">
            Search signs
          </label>
          <input
            id="dictionary-search"
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search signs..."
            className="min-h-[44px] w-full rounded-lg border border-ink/15 bg-white py-2.5 pl-11 pr-4 text-base"
          />
        </div>

        <div
          className="flex flex-wrap gap-2"
          role="group"
          aria-label="Filter by category"
        >
          {CATEGORIES.map((category) => {
            const active = activeCategories.includes(category);
            const count = KSL_VOCABULARY.filter(
              (entry) => entry.category === category
            ).length;
            if (count === 0) return null;
            return (
              <button
                key={category}
                type="button"
                aria-pressed={active}
                onClick={() => toggleCategory(category)}
                className={`min-h-[44px] rounded-lg border px-4 text-sm font-semibold ${
                  active
                    ? "border-ink bg-ink text-paper"
                    : "border-ink/15 text-ink-soft hover:text-ink"
                }`}
              >
                {category}
                <span className="ml-1.5 font-normal opacity-70">{count}</span>
              </button>
            );
          })}
        </div>
      </div>

      <p className="mt-6 text-sm text-ink-soft" aria-live="polite">
        {results.length} of {KSL_VOCABULARY.length} signs shown.
      </p>

      <div className="mt-4">
        {results.length === 0 ? (
          <EmptyState
            icon={<Search size={28} />}
            title="No signs match that search"
            description="Try a different word, or clear the category filters. Fadhili's verified KSL vocabulary is still small."
          />
        ) : (
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {results.map((entry) => {
              const canRecognize = recognizable?.has(entry.id) ?? false;
              return (
                <li key={entry.id}>
                  <Card className="flex h-full flex-col gap-2">
                    <div className="flex items-start justify-between gap-3">
                      <h2 className="text-lg font-medium">{entry.gloss}</h2>
                      <span className="shrink-0 rounded bg-ink/5 px-2 py-1 text-xs font-semibold text-ink-soft">
                        {entry.category}
                      </span>
                    </div>
                    <p className="text-sm text-ink-soft">{entry.description}</p>
                    {canRecognize && (
                      <span className="mt-auto inline-flex items-center gap-1.5 self-start rounded bg-teal-100 px-2 py-1 text-xs font-semibold text-teal-800">
                        <Camera size={12} aria-hidden="true" />
                        Recognised on camera
                      </span>
                    )}
                  </Card>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <p className="mt-12 max-w-3xl border-t border-ink/10 pt-5 text-sm text-ink-soft">
        Definitions describe the English word. They are not descriptions of how
        the sign is formed, and they do not cover regional variation within
        Kenyan Sign Language — the source dataset documents neither, and
        Fadhili will not invent them.
      </p>
    </div>
  );
}
