import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import {
  Video,
  VideoOff,
  Upload,
  Copy,
  Volume2,
  Trash2,
  AlertCircle,
} from "lucide-react";
import { useCamera } from "../hooks/useCamera";
import { useKSLRecognition } from "../hooks/useKSLRecognition";
import { useSignLanguage } from "../context/LanguageContext";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { StatusBadge } from "../components/ui/StatusBadge";
import { checkHealth } from "../services/api";
import type { KSLPredictionCandidate } from "../services/api";
import type { RecognitionStatus } from "../types";

const UPLOAD_NOT_CONNECTED_MESSAGE = "Recognition model not connected";

interface RecognitionPanelData {
  statusKind: RecognitionStatus;
  statusLabel: string;
  detected: string;
  confidenceText: string;
  translation: string;
  candidates: KSLPredictionCandidate[];
}

const SPEECH_COOLDOWN_MS = 2500;

export default function Interpreter() {
  const { language } = useSignLanguage();

  const {
    videoRef,
    permission,
    isActive,
    errorMessage,
    startCamera,
    stopCamera,
  } = useCamera();

  const [mode, setMode] = useState<"camera" | "upload">("camera");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadUrl, setUploadUrl] = useState<string | null>(null);
  const [backendWarning, setBackendWarning] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const isCameraMode = mode === "camera";

  const kslRecognition = useKSLRecognition(
    videoRef,
    language,
    isCameraMode && isActive
  );

  const lastSpokenRef = useRef<string | null>(null);
  const lastSpeakTimeRef = useRef(0);

  // Check the FastAPI backend and model once when the page loads.
  useEffect(() => {
    let cancelled = false;

    const checkBackend = async () => {
      const result = await checkHealth();

      if (cancelled) return;

      if (result.ok === false) {
        setBackendWarning(`Backend unavailable: ${result.error}`);
        return;
      }

      if (!result.data.model_loaded) {
        setBackendWarning(
          "The KSL recognition model failed to load on the backend."
        );
        return;
      }

      setBackendWarning(null);
    };

    void checkBackend();

    return () => {
      cancelled = true;
    };
  }, []);

  // Automatically speak newly accepted predictions while avoiding
  // repeatedly speaking the same held sign.
  useEffect(() => {
    if (!isCameraMode || !kslRecognition.isSupportedLanguage) return;

    const prediction = kslRecognition.lastPrediction;

    if (
      !prediction ||
      !prediction.accepted ||
      !prediction.prediction
    ) {
      return;
    }

    const label = prediction.prediction;
    const now = performance.now();

    if (label === lastSpokenRef.current) return;

    if (
      now - lastSpeakTimeRef.current <
      SPEECH_COOLDOWN_MS
    ) {
      return;
    }

    lastSpokenRef.current = label;
    lastSpeakTimeRef.current = now;

    const utterance = new SpeechSynthesisUtterance(label);
    window.speechSynthesis.speak(utterance);
  }, [
    isCameraMode,
    kslRecognition.isSupportedLanguage,
    kslRecognition.lastPrediction,
  ]);

  const handleFileChange = (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];

    if (!file) return;

    if (uploadUrl) {
      URL.revokeObjectURL(uploadUrl);
    }

    setUploadedFile(file);
    setUploadUrl(URL.createObjectURL(file));
  };

  const clearUpload = () => {
    if (uploadUrl) {
      URL.revokeObjectURL(uploadUrl);
    }

    setUploadedFile(null);
    setUploadUrl(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const uploadPanel: RecognitionPanelData = {
    statusKind: "unavailable",
    statusLabel: UPLOAD_NOT_CONNECTED_MESSAGE,
    detected: "—",
    confidenceText: "—",
    translation:
      "Uploaded-video recognition isn't connected yet — use live camera for KSL recognition.",
    candidates: [],
  };

  function buildCameraPanel(): RecognitionPanelData {
    if (!kslRecognition.isSupportedLanguage) {
      return {
        statusKind: "unavailable",
        statusLabel:
          kslRecognition.comingSoonMessage ?? "Coming soon",
        detected: "—",
        confidenceText: "—",
        translation:
          kslRecognition.comingSoonMessage ?? "Coming soon",
        candidates: [],
      };
    }

    if (!isActive) {
      return {
        statusKind: "idle",
        statusLabel: "Camera is off",
        detected: "—",
        confidenceText: "—",
        translation:
          "Start the camera to begin KSL recognition.",
        candidates: [],
      };
    }

    if (kslRecognition.isWarmingUp) {
      return {
        statusKind: "loading",
        statusLabel: `Preparing recognition: ${kslRecognition.framesCollected}/30`,
        detected: "—",
        confidenceText: "—",
        translation: "Collecting frames…",
        candidates: [],
      };
    }

    if (kslRecognition.lastError) {
      return {
        statusKind: "error",
        statusLabel: kslRecognition.lastError,
        detected: "—",
        confidenceText: "—",
        translation: "Recognition is temporarily unavailable.",
        candidates: [],
      };
    }

    const prediction = kslRecognition.lastPrediction;

    if (!prediction) {
      return {
        statusKind: "loading",
        statusLabel: kslRecognition.isPredicting
          ? "Recognizing…"
          : "Ready",
        detected: "—",
        confidenceText: "—",
        translation:
          "Hold a sign steady in view of the camera.",
        candidates: [],
      };
    }

    if (!prediction.accepted) {
      return {
        statusKind: "idle",
        statusLabel: "No confident recognition",
        detected: "No confident recognition",
        confidenceText: `${Math.round(
          prediction.confidence * 100
        )}% (below threshold)`,
        translation: "No confident recognition",
        candidates: prediction.top_candidates,
      };
    }

    return {
      statusKind: "success",
      statusLabel: "Experimental KSL recognition",
      detected: prediction.prediction ?? "—",
      confidenceText: `${Math.round(
        prediction.confidence * 100
      )}%`,
      translation: prediction.prediction ?? "—",
      candidates: prediction.top_candidates,
    };
  }

  const panel = isCameraMode
    ? buildCameraPanel()
    : uploadPanel;

  const speakableText =
    isCameraMode &&
    kslRecognition.lastPrediction?.accepted
      ? kslRecognition.lastPrediction.prediction
      : null;

  const handleSpeak = () => {
    if (!speakableText) return;

    const utterance =
      new SpeechSynthesisUtterance(speakableText);

    window.speechSynthesis.speak(utterance);
  };

  const handleCopy = async () => {
    if (!speakableText) return;

    await navigator.clipboard.writeText(speakableText);
  };

  const handleClear = () => {
    kslRecognition.resetRecognition();

    lastSpokenRef.current = null;
    lastSpeakTimeRef.current = 0;
  };

  return (
    <div className="mx-auto max-w-6xl px-5 py-12">
      <header className="max-w-2xl">
        <h1 className="text-3xl font-medium sm:text-4xl">
          Interpreter
        </h1>

        <p className="mt-3 text-ink-soft">
          Use your camera to interpret {language} sign
          language. KSL recognition runs live and is
          experimental — ASL and BSL recognition are coming
          soon.
        </p>
      </header>

      {backendWarning && (
        <div
          role="alert"
          className="mt-6 flex items-start gap-2 rounded-xl2 border border-ochre-300/60 bg-ochre-100/60 p-4 text-sm text-ink"
        >
          <AlertCircle
            size={18}
            className="mt-0.5 shrink-0"
            aria-hidden="true"
          />

          <span>{backendWarning}</span>
        </div>
      )}

      <div
        className="mt-8 flex gap-2"
        role="tablist"
        aria-label="Interpreter input mode"
      >
        <button
          role="tab"
          aria-selected={mode === "camera"}
          onClick={() => setMode("camera")}
          className={`min-h-[44px] rounded-full px-4 text-sm font-semibold ${
            mode === "camera"
              ? "bg-ink text-paper"
              : "border border-ink/15 text-ink-soft"
          }`}
        >
          Live camera
        </button>

        <button
          role="tab"
          aria-selected={mode === "upload"}
          onClick={() => setMode("upload")}
          className={`min-h-[44px] rounded-full px-4 text-sm font-semibold ${
            mode === "upload"
              ? "bg-ink text-paper"
              : "border border-ink/15 text-ink-soft"
          }`}
        >
          Upload video
        </button>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.2fr_1fr]">
        <Card
          padded={false}
          className="overflow-hidden"
        >
          {mode === "camera" ? (
            <div className="relative aspect-video bg-ink">
              <video
                ref={videoRef}
                playsInline
                muted
                aria-label="Live camera preview"
                className={`h-full w-full object-cover ${
                  isActive ? "" : "hidden"
                }`}
              />

              {!isActive && (
                <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center text-paper/80">
                  <Video
                    size={32}
                    aria-hidden="true"
                  />

                  <p className="text-sm">
                    {permission === "requesting"
                      ? "Requesting camera access…"
                      : "Camera is off. Start it to begin interpreting."}
                  </p>
                </div>
              )}

              {isActive && (
                <span className="absolute left-4 top-4 inline-flex items-center gap-2 rounded-full bg-ink/80 px-3 py-1 text-xs font-semibold text-paper">
                  <span
                    className="h-2 w-2 animate-pulse rounded-full bg-signal-red motion-reduce:animate-none"
                    aria-hidden="true"
                  />

                  Camera active
                </span>
              )}
            </div>
          ) : (
            <div className="p-6">
              {uploadUrl ? (
                <video
                  src={uploadUrl}
                  controls
                  aria-label="Uploaded video preview"
                  className="aspect-video w-full rounded-lg bg-ink object-contain"
                />
              ) : (
                <div className="flex aspect-video flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-ink/20 text-center text-ink-soft">
                  <Upload
                    size={28}
                    aria-hidden="true"
                  />

                  <p className="text-sm">
                    No video uploaded yet
                  </p>
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="sr-only"
                id="video-upload"
              />
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3 border-t border-ink/10 p-4">
            {mode === "camera" ? (
              isActive ? (
                <Button
                  variant="secondary"
                  icon={<VideoOff size={16} />}
                  onClick={stopCamera}
                >
                  Stop Camera
                </Button>
              ) : (
                <Button
                  icon={<Video size={16} />}
                  onClick={startCamera}
                  disabled={permission === "requesting"}
                >
                  Start Camera
                </Button>
              )
            ) : (
              <>
                <Button
                  icon={<Upload size={16} />}
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                >
                  Upload Video
                </Button>

                {uploadedFile && (
                  <Button
                    variant="secondary"
                    icon={<Trash2 size={16} />}
                    onClick={clearUpload}
                  >
                    Remove
                  </Button>
                )}
              </>
            )}
          </div>

          {errorMessage && (
            <div
              role="alert"
              className="flex items-start gap-2 border-t border-ink/10 bg-signal-red/10 p-4 text-sm text-signal-red"
            >
              <AlertCircle
                size={18}
                className="mt-0.5 shrink-0"
                aria-hidden="true"
              />

              <span>{errorMessage}</span>
            </div>
          )}
        </Card>

        <Card className="flex flex-col gap-5">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium">
              Recognition
            </h2>

            <StatusBadge
              status={panel.statusKind}
              label={panel.statusLabel}
            />
          </div>

          {isCameraMode &&
            kslRecognition.isSupportedLanguage && (
              <p className="text-xs text-ink-soft">
                Experimental KSL recognition — recognition
                may be inaccurate for unfamiliar signers.
                Model confidence is not a measure of
                accuracy.
              </p>
            )}

          <dl className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <dt className="text-ink-soft">
                Detected sign
              </dt>

              <dd className="mt-1 font-semibold text-ink">
                {panel.detected}
              </dd>
            </div>

            <div>
              <dt className="text-ink-soft">
                Model confidence
              </dt>

              <dd className="mt-1 font-semibold text-ink">
                {panel.confidenceText}
              </dd>
            </div>

            <div className="col-span-2">
              <dt className="text-ink-soft">
                Translation
              </dt>

              <dd className="mt-1 min-h-[2.5rem] rounded-lg bg-ink/[0.03] p-3 text-ink">
                {panel.translation}
              </dd>
            </div>
          </dl>

          {panel.candidates.length > 0 && (
            <div>
              <p className="mb-2 text-sm text-ink-soft">
                Top candidates
              </p>

              <ul className="flex flex-col gap-1 text-sm">
                {panel.candidates.map((candidate) => (
                  <li
                    key={candidate.label}
                    className="flex items-center justify-between rounded-lg bg-ink/[0.03] px-3 py-2"
                  >
                    <span className="font-medium text-ink">
                      {candidate.label}
                    </span>

                    <span className="text-ink-soft">
                      {Math.round(
                        candidate.confidence * 100
                      )}
                      %
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            <Button
              variant="secondary"
              size="md"
              icon={<Volume2 size={16} />}
              onClick={handleSpeak}
              disabled={!speakableText}
            >
              Speak
            </Button>

            <Button
              variant="secondary"
              size="md"
              icon={<Copy size={16} />}
              onClick={handleCopy}
              disabled={!speakableText}
            >
              Copy
            </Button>

            <Button
              variant="ghost"
              size="md"
              icon={<Trash2 size={16} />}
              onClick={handleClear}
              disabled={
                !speakableText &&
                !kslRecognition.lastPrediction
              }
            >
              Clear
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}