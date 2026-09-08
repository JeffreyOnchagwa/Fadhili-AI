import { useEffect, useMemo, useState } from "react";
import { Check, HelpCircle } from "lucide-react";
import { translateTextToKSL } from "../../data/kslVocabulary";
import { getKSLClasses } from "../../services/api";

interface Props {
  text: string;
}

/**
 * Renders typed or spoken text as verified Kenyan Sign Language signs.
 *
 * Honesty constraints, which are the point of this component:
 *
 * - It only ever shows signs backed by recorded KSL data. Words outside
 *   the verified vocabulary are named explicitly as unavailable rather
 *   than silently dropped or approximated.
 * - It does not claim to produce grammatical KSL. Word-by-word lookup
 *   is not translation: KSL has its own grammar, and signed order,
 *   facial grammar and classifiers are not modelled here at all. The
 *   UI states this rather than letting the layout imply fluency.
 */
export function TextToSignOutput({ text }: Props) {
  const tokens = useMemo(() => translateTextToKSL(text), [text]);
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

  if (tokens.length === 0) {
    return (
      <div className="rounded-xl2 border border-dashed border-ink/20 p-8 text-center">
        <p className="text-sm text-ink-soft">
          Type or speak something to see it mapped to verified KSL signs.
        </p>
      </div>
    );
  }

  const matched = tokens.filter((token) => token.entry !== null);
  const unmatched = tokens.filter((token) => token.entry === null);

  return (
    <div className="flex flex-col gap-5">
      <ul className="flex flex-wrap gap-2" aria-label="Signs for your text">
        {tokens.map((token, index) => {
          const entry = token.entry;
          const known = entry !== null;
          return (
            <li
              key={`${token.original}-${index}`}
              className={`inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm ${
                known
                  ? "border-teal-600/30 bg-teal-50 text-teal-900"
                  : "border-ink/15 bg-ink/[0.03] text-ink-soft"
              }`}
            >
              {known ? (
                <Check size={14} className="text-teal-600" aria-hidden="true" />
              ) : (
                <HelpCircle size={14} aria-hidden="true" />
              )}
              <span className="font-semibold">{token.original}</span>
              <span className="sr-only">
                {known
                  ? `has a verified KSL sign`
                  : `has no verified KSL sign available`}
              </span>
            </li>
          );
        })}
      </ul>

      <p className="text-sm text-ink-soft">
        {matched.length} of {tokens.length}{" "}
        {tokens.length === 1 ? "word has" : "words have"} a verified KSL sign in
        Fadhili&rsquo;s vocabulary.
      </p>

      {unmatched.length > 0 && (
        <div className="rounded-lg border border-ochre-300/60 bg-ochre-100/60 p-3">
          <p className="text-sm text-ochre-800">
            No verified sign is available for{" "}
            <span className="font-semibold">
              {unmatched.map((token) => token.original).join(", ")}
            </span>
            . Fadhili will not invent a sign, so these are left untranslated.
          </p>
        </div>
      )}

      {recognizable !== null && matched.length > 0 && (
        <p className="text-xs text-ink-soft">
          {
            matched.filter((token) => recognizable.has(token.entry!.id)).length
          }{" "}
          of these can also be recognised from your camera by the current
          model.
        </p>
      )}

      <p className="border-t border-ink/10 pt-4 text-xs text-ink-soft">
        This is a word-by-word vocabulary lookup, not a translation. Kenyan
        Sign Language has its own grammar, and meaning is also carried by
        facial expression and body movement that a word list cannot
        represent. For anything that matters, work with a qualified KSL
        interpreter.
      </p>
    </div>
  );
}
