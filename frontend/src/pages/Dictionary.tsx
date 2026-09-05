import { useMemo, useState } from "react";
import { Search, BookOpen } from "lucide-react";
import { dictionaryEntries } from "../data/dictionary";
import { SIGN_LANGUAGES } from "../types";
import type { SupportedSignLanguage } from "../types";
import { Card } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";

export default function Dictionary() {
  const [query, setQuery] = useState("");
  const [activeFilters, setActiveFilters] = useState<SupportedSignLanguage[]>([]);

  const results = useMemo(() => {
    return dictionaryEntries.filter((entry) => {
      const matchesQuery = entry.word.toLowerCase().includes(query.toLowerCase());
      const matchesFilter =
        activeFilters.length === 0 || activeFilters.includes(entry.language);
      return matchesQuery && matchesFilter;
    });
  }, [query, activeFilters]);

  const toggleFilter = (code: SupportedSignLanguage) => {
    setActiveFilters((prev) =>
      prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code]
    );
  };

  return (
    <div className="mx-auto max-w-6xl px-5 py-12">
      <header className="max-w-2xl">
        <h1 className="text-3xl font-medium sm:text-4xl">Dictionary</h1>
        <p className="mt-3 text-ink-soft">
          Look up signs across KSL, ASL, and BSL. Entries are added only once
          they've been verified — nothing here is guessed or generated.
        </p>
      </header>

      <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:items-center">
        <div className="relative flex-1">
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
            className="min-h-[44px] w-full rounded-full border border-ink/15 bg-white py-2.5 pl-11 pr-4 text-base"
          />
        </div>
        <div className="flex gap-2" role="group" aria-label="Filter by sign language">
          {SIGN_LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              type="button"
              aria-pressed={activeFilters.includes(lang.code)}
              onClick={() => toggleFilter(lang.code)}
              className={`min-h-[44px] rounded-full border px-4 text-sm font-semibold ${
                activeFilters.includes(lang.code)
                  ? "border-ink bg-ink text-paper"
                  : "border-ink/15 text-ink-soft"
              }`}
            >
              {lang.flag} {lang.code}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-8">
        {results.length === 0 ? (
          <EmptyState
            icon={<BookOpen size={28} />}
            title="No verified entries yet"
            description="The dictionary is built and ready — verified KSL, ASL, and BSL entries will appear here as they're added, each with a source and usage example."
          />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {results.map((entry) => (
              <Card key={entry.id}>
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-medium">{entry.word}</h3>
                  <span className="rounded-full bg-ink/5 px-2.5 py-1 text-xs font-semibold text-ink-soft">
                    {entry.language}
                  </span>
                </div>
                <p className="mt-2 text-sm text-ink-soft">{entry.meaning}</p>
                {entry.usageExample && (
                  <p className="mt-2 text-sm italic text-ink-soft">"{entry.usageExample}"</p>
                )}
                {entry.source && (
                  <p className="mt-3 text-xs text-ink-soft/70">Source: {entry.source}</p>
                )}
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
