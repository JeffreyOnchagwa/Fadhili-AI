import { useEffect, useRef, useState } from "react";
import { Keyboard, Mic, Play, Pause, RotateCcw, Gauge } from "lucide-react";
import { useSignLanguage } from "../context/LanguageContext";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { EmptyState } from "../components/ui/EmptyState";
import { SIGN_LANGUAGES } from "../types";

type SpeechRecognitionCtor = new () => SpeechRecognition;

function getSpeechRecognitionCtor(): SpeechRecognitionCtor | null {
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export default function Translate() {
  const { language, setLanguage } = useSignLanguage();
  const [mode, setMode] = useState<"type" | "mic">("type");
  const [text, setText] = useState("");
  const [listening, setListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  useEffect(() => {
    setSpeechSupported(Boolean(getSpeechRecognitionCtor()));
  }, []);

  const startListening = () => {
    const Ctor = getSpeechRecognitionCtor();
    if (!Ctor) {
      setSpeechSupported(false);
      return;
    }
    const recognition = new Ctor();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0]?.transcript ?? "")
        .join(" ");
      setText(transcript);
    };
    recognition.onend = () => setListening(false);
    recognition.onerror = () => setListening(false);
    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  };

  const stopListening = () => {
    recognitionRef.current?.stop();
    setListening(false);
  };

  useEffect(() => {
    return () => recognitionRef.current?.stop();
  }, []);

  const hasContent = text.trim().length > 0;

  return (
    <div className="mx-auto max-w-6xl px-5 py-12">
      <header className="max-w-2xl">
        <h1 className="text-3xl font-medium sm:text-4xl">Translate</h1>
        <p className="mt-3 text-ink-soft">
          Type or speak, and Fadhili AI will prepare it for sign output. Sign
          generation isn't connected yet, so the player below will show an
          honest "not connected" state.
        </p>
      </header>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <Card className="flex flex-col gap-4">
          <div className="flex gap-2" role="tablist" aria-label="Input mode">
            <button
              role="tab"
              aria-selected={mode === "type"}
              onClick={() => setMode("type")}
              className={`inline-flex min-h-[44px] items-center gap-2 rounded-full px-4 text-sm font-semibold ${
                mode === "type" ? "bg-ink text-paper" : "border border-ink/15 text-ink-soft"
              }`}
            >
              <Keyboard size={16} aria-hidden="true" /> Type
            </button>
            <button
              role="tab"
              aria-selected={mode === "mic"}
              onClick={() => setMode("mic")}
              className={`inline-flex min-h-[44px] items-center gap-2 rounded-full px-4 text-sm font-semibold ${
                mode === "mic" ? "bg-ink text-paper" : "border border-ink/15 text-ink-soft"
              }`}
            >
              <Mic size={16} aria-hidden="true" /> Microphone
            </button>
          </div>

          <label htmlFor="translate-input" className="sr-only">
            Text to translate into sign language
          </label>
          <textarea
            id="translate-input"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Type something to translate into sign language..."
            rows={7}
            className="w-full resize-none rounded-xl2 border border-ink/15 bg-white p-4 text-base text-ink placeholder:text-ink-soft/70"
          />

          {mode === "mic" && (
            <div>
              {speechSupported ? (
                <Button
                  variant={listening ? "danger" : "primary"}
                  icon={<Mic size={16} />}
                  onClick={listening ? stopListening : startListening}
                >
                  {listening ? "Stop listening" : "Start listening"}
                </Button>
              ) : (
                <p className="rounded-lg bg-ochre-100 p-3 text-sm text-ochre-700">
                  Speech recognition isn't supported in this browser. Try
                  Chrome or Edge, or switch to typing instead.
                </p>
              )}
            </div>
          )}

          <div className="flex items-center gap-3 border-t border-ink/10 pt-4">
            <label htmlFor="output-language" className="text-sm font-semibold text-ink-soft">
              Output language
            </label>
            <select
              id="output-language"
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
        </Card>

        <Card className="flex flex-col gap-5">
          <h2 className="text-lg font-medium">Sign output</h2>
          <EmptyState
            title="Sign generation model not connected"
            description={
              hasContent
                ? "Your text is ready to send once the sign-generation backend is connected."
                : "Type or speak something to see it prepared for sign output."
            }
          />
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" icon={<Play size={16} />} disabled>
              Play
            </Button>
            <Button variant="secondary" icon={<Pause size={16} />} disabled>
              Pause
            </Button>
            <Button variant="secondary" icon={<RotateCcw size={16} />} disabled>
              Repeat
            </Button>
            <Button variant="secondary" icon={<Gauge size={16} />} disabled>
              Slow Down
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}
