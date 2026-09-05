import { useState, useRef, useEffect } from "react";
import { ChevronDown, Check } from "lucide-react";
import { useSignLanguage } from "../../context/LanguageContext";
import { SIGN_LANGUAGES } from "../../types";

export function LanguageSelector() {
  const { language, setLanguage } = useSignLanguage();
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const current = SIGN_LANGUAGES.find((l) => l.code === language)!;

  useEffect(() => {
    function handleClick(event: MouseEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div className="relative" ref={containerRef}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="inline-flex min-h-[44px] items-center gap-2 rounded-full border border-ink/15 bg-white px-4 py-2 text-sm font-semibold text-ink hover:border-ink/30"
      >
        <span aria-hidden="true">{current.flag}</span>
        <span>{current.code}</span>
        <ChevronDown size={16} className="text-ink-soft" aria-hidden="true" />
      </button>
      {open && (
        <ul
          role="listbox"
          aria-label="Choose a sign language"
          className="absolute right-0 z-20 mt-2 w-56 overflow-hidden rounded-xl2 border border-ink/10 bg-white shadow-soft"
        >
          {SIGN_LANGUAGES.map((lang) => (
            <li key={lang.code} role="option" aria-selected={lang.code === language}>
              <button
                type="button"
                onClick={() => {
                  setLanguage(lang.code);
                  setOpen(false);
                }}
                className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm hover:bg-ink/[0.03]"
              >
                <span aria-hidden="true">{lang.flag}</span>
                <span className="flex-1">
                  <span className="block font-semibold text-ink">{lang.code}</span>
                  <span className="block text-xs text-ink-soft">{lang.label}</span>
                </span>
                {lang.code === language && (
                  <Check size={16} className="text-teal-500" aria-hidden="true" />
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
