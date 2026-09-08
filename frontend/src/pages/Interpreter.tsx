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
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { StatusBadge } from "../components/ui/StatusBadge";
import { LandmarkOverlay } from "../components/interpreter/LandmarkOverlay";
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
  const [speechEnabled, setSpeechEnabled] = useState(false);
  // On by default: it's the clearest way to see why recognition might
  // be struggling (a hand dropping out of view reads very differently
  // from "the model doesn't know this sign").
  const [showSkeleton, setShowSkeleton] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const isCameraMode = mode === "camera";

  const kslRecognition = useKSLRecognition(
    videoRef,
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

  // Speak a sign only once it has settled across several inferences,
  // and only when speech is switched on. Keying off `sequence` (which
  // increments only when the settled sign CHANGES) means holding one
  // sign speaks it once rather than once per inference cycle.
  useEffect(() => {
    if (!isCameraMode || !speechEnabled) return;

    const settled = kslRecognition.stableSign;
    if (!settled) return;

    const spokenKey = `${settled.sequence}:${settled.label}`;
    if (spokenKey === lastSpokenRef.current) return;

    const now = performance.now();
    if (now - lastSpeakTimeRef.current < SPEECH_COOLDOWN_MS) return;

    lastSpokenRef.current = spokenKey;
    lastSpeakTimeRef.current = now;

    const utterance = new SpeechSynthesisUtterance(settled.label);
    window.speechSynthesis.speak(utterance);
  }, [isCameraMode, speechEnabled, kslRecognition.stableSign]);

  // Stop any queued speech when speech is turned off or the camera
  // stops, so a toggle takes effect immediately.
  useEffect(() => {
    if (!speechEnabled || !isCameraMode) {
      window.speechSynthesis.cancel();
    }
  }, [speechEnabled, isCameraMode]);

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
        statusLabel: "Starting up",
        detected: "—",
        confidenceText: "—",
        translation: "Finding you in the camera…",
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
    const settled = kslRecognition.stableSign;

    if (!prediction) {
      return {
        statusKind: "loading",
        statusLabel: kslRecognition.isPredicting ? "Reading…" : "Ready",
        detected: "—",
        confidenceText: "—",
        translation: "Hold a sign steady in view of the camera.",
        candidates: [],
      };
    }

    // The server's motion gate decided nobody is signing. Say that
    // plainly instead of leaving a stale word on screen.
    if (kslRecognition.phase === "no_sign") {
      return {
        statusKind: "idle",
        statusLabel: "No sign detected",
        detected: "—",
        confidenceText: "—",
        translation: "Start signing and Fadhili will read it.",
        candidates: [],
      };
    }

    if (prediction.reason === "no_person_detected") {
      return {
        statusKind: "idle",
        statusLabel: "Nobody in view",
        detected: "—",
        confidenceText: "—",
        translation:
          "Move so your head, shoulders and hands are all in frame.",
        candidates: [],
      };
    }

    // A sign is only presented once it has held across several
    // readings. Until then the panel says it is still watching rather
    // than flashing whichever label the latest inference produced.
    if (settled && kslRecognition.phase === "settled") {
      return {
        statusKind: "success",
        statusLabel: `Held across ${settled.agreement} of 5 readings`,
        detected: settled.label,
        confidenceText: `${Math.round(settled.meanConfidence * 100)}%`,
        translation: settled.label,
        candidates: prediction.top_candidates,
      };
    }

    if (kslRecognition.phase === "uncertain") {
      return {
        statusKind: "idle",
        statusLabel: "Reading not settled",
        detected: "—",
        confidenceText: "—",
        translation:
          "The reading is still changing. Hold the sign steady.",
        candidates: prediction.top_candidates,
      };
    }

    return {
      statusKind: "idle",
      statusLabel: "No confident recognition",
      detected: "—",
      confidenceText: `${Math.round(prediction.confidence * 100)}% (below threshold)`,
      translation: "No confident recognition",
      candidates: prediction.top_candidates,
    };
  }

  const panel = isCameraMode ? buildCameraPanel() : uploadPanel;

  const speakableText =
    isCameraMode && kslRecognition.phase === "settled"
      ? kslRecognition.stableSign?.label ?? null
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
          Use your camera to read Kenyan Sign Language. Recognition runs
          live on your device and is experimental: it covers a small
          vocabulary and can be confidently wrong, particularly for people
          and settings unlike those it was trained on.
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
          className={`min-h-[44px] rounded-lg px-4 text-sm font-semibold ${
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
          className={`min-h-[44px] rounded-lg px-4 text-sm font-semibold ${
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

              <LandmarkOverlay
                videoRef={videoRef}
                landmarksRef={kslRecognition.landmarksRef}
                active={isActive && showSkeleton}
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

              {isActive && (
                <button
                  type="button"
                  onClick={() => setShowSkeleton((on) => !on)}
                  aria-pressed={showSkeleton}
                  className="absolute right-4 top-4 inline-flex min-h-[36px] items-center gap-2 rounded-lg bg-ink/80 px-3 py-1 text-xs font-semibold text-paper hover:bg-ink"
                >
                  {showSkeleton ? "Hide tracking" : "Show tracking"}
                </button>
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

          {isCameraMode && (
            <p className="text-xs text-ink-soft">
              Experimental KSL recognition. The percentage shown is the
              model&rsquo;s own confidence, which is not calibrated and
              is not a measure of accuracy — it stays high even when the
              reading is wrong.
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

          {isCameraMode && (
            <div className="flex items-center justify-between gap-4 rounded-lg border border-ink/10 bg-ink/[0.02] p-3">
              <label
                htmlFor="speak-automatically"
                className="text-sm font-semibold text-ink"
              >
                Speak signs aloud
                <span className="block text-xs font-normal text-ink-soft">
                  Reads each sign once it settles.
                </span>
              </label>
              <button
                id="speak-automatically"
                type="button"
                role="switch"
                aria-checked={speechEnabled}
                onClick={() => setSpeechEnabled((on) => !on)}
                className={`relative inline-flex h-7 w-12 shrink-0 items-center rounded-full border transition-colors ${
                  speechEnabled
                    ? "border-teal-700 bg-teal-600"
                    : "border-ink/25 bg-ink/10"
                }`}
              >
                <span
                  className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
                    speechEnabled ? "translate-x-6" : "translate-x-1"
                  }`}
                />
              </button>
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