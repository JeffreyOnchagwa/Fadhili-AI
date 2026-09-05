import { createContext, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { SupportedSignLanguage } from "../types";

const STORAGE_KEY = "fadhili:sign-language";

interface LanguageContextValue {
  language: SupportedSignLanguage;
  setLanguage: (language: SupportedSignLanguage) => void;
}

const LanguageContext = createContext<LanguageContextValue | undefined>(
  undefined
);

function readInitialLanguage(): SupportedSignLanguage {
  if (typeof window === "undefined") return "KSL";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === "KSL" || stored === "ASL" || stored === "BSL") return stored;
  return "KSL";
}

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] =
    useState<SupportedSignLanguage>(readInitialLanguage);

  const setLanguage = (next: SupportedSignLanguage) => {
    setLanguageState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Storage may be unavailable (private browsing, etc.) — the
      // selection still works for the current session.
    }
  };

  const value = useMemo(() => ({ language, setLanguage }), [language]);

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useSignLanguage(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useSignLanguage must be used within a LanguageProvider");
  }
  return ctx;
}
