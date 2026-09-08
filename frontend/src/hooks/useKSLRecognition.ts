import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

import { predictKSL } from "../services/api";
import type { KSLPredictionResponse } from "../services/api";
import { extractFeatures, SEQUENCE_LENGTH } from "../services/mediapipeFeatures";
import {
  loadMediaPipeHolistic,
  locateHolisticFile,
} from "../services/mediapipeHolisticLoader";
import type { HolisticInstance } from "../services/mediapipeHolisticLoader";

const PREDICTION_INTERVAL_MS = 800;
const FEATURE_LENGTH = 150;

/**
 * Temporal stabilization.
 *
 * The model runs roughly once a second and is substantially
 * overconfident: on held-out signers it averaged 0.92 confidence on
 * predictions that were WRONG. Confidence alone is therefore a poor
 * gate, and showing every raw inference makes the UI flicker between
 * unrelated words.
 *
 * Agreement over time is the stronger signal, so a sign is only
 * surfaced as settled once the same label wins a majority of a short
 * rolling window. This cannot make the model correct — it only stops
 * the interface from presenting noise as if it were a reading.
 */
const HISTORY_WINDOW = 5;
const AGREEMENT_REQUIRED = 3;

export type RecognitionPhase =
  | "idle"
  | "collecting"
  | "watching"
  | "uncertain"
  | "settled";

export interface StableSign {
  label: string;
  /** Mean confidence across the agreeing predictions in the window. */
  meanConfidence: number;
  /** How many of the last HISTORY_WINDOW predictions agreed. */
  agreement: number;
  /** Increments each time a NEW sign settles; drives speech. */
  sequence: number;
}

export interface UseKSLRecognitionResult {
  framesCollected: number;
  isWarmingUp: boolean;
  isPredicting: boolean;
  phase: RecognitionPhase;
  /** Raw latest inference. Useful for diagnostics, not for display. */
  lastPrediction: KSLPredictionResponse | null;
  /** The settled sign, or null while nothing has stabilized. */
  stableSign: StableSign | null;
  lastError: string | null;
  resetRecognition: () => void;
}

export function useKSLRecognition(
  videoRef: RefObject<HTMLVideoElement | null>,
  isCameraActive: boolean
): UseKSLRecognitionResult {
  const [framesCollected, setFramesCollected] = useState(0);
  const [isPredicting, setIsPredicting] = useState(false);
  const [lastPrediction, setLastPrediction] =
    useState<KSLPredictionResponse | null>(null);
  const [stableSign, setStableSign] = useState<StableSign | null>(null);
  const [phase, setPhase] = useState<RecognitionPhase>("idle");
  const [lastError, setLastError] = useState<string | null>(null);

  const bufferRef = useRef<number[][]>([]);
  const historyRef = useRef<{ label: string | null; confidence: number }[]>([]);
  const settledLabelRef = useRef<string | null>(null);
  const sequenceRef = useRef(0);
  const isPredictingRef = useRef(false);
  const lastPredictTimeRef = useRef(0);
  const activeRef = useRef(false);

  const resetRecognition = useCallback(() => {
    bufferRef.current = [];
    historyRef.current = [];
    settledLabelRef.current = null;
    lastPredictTimeRef.current = 0;
    setFramesCollected(0);
    setLastPrediction(null);
    setStableSign(null);
    setLastError(null);
    setPhase(activeRef.current ? "collecting" : "idle");
  }, []);

  /**
   * Folds one inference into the rolling window and decides whether a
   * sign has settled.
   */
  const integrate = useCallback((response: KSLPredictionResponse) => {
    const history = historyRef.current;
    history.push({
      label: response.accepted ? response.prediction : null,
      confidence: response.confidence,
    });
    if (history.length > HISTORY_WINDOW) history.shift();

    // Tally accepted labels across the window.
    const tally = new Map<string, number[]>();
    for (const item of history) {
      if (!item.label) continue;
      const scores = tally.get(item.label) ?? [];
      scores.push(item.confidence);
      tally.set(item.label, scores);
    }

    let winner: string | null = null;
    let winnerScores: number[] = [];
    for (const [label, scores] of tally) {
      if (scores.length > winnerScores.length) {
        winner = label;
        winnerScores = scores;
      }
    }

    if (winner && winnerScores.length >= AGREEMENT_REQUIRED) {
      const meanConfidence =
        winnerScores.reduce((sum, value) => sum + value, 0) /
        winnerScores.length;

      // Only bump the sequence when the settled sign actually changes,
      // so speech and announcements fire once per sign rather than once
      // per inference.
      if (settledLabelRef.current !== winner) {
        settledLabelRef.current = winner;
        sequenceRef.current += 1;
      }

      setStableSign({
        label: winner,
        meanConfidence,
        agreement: winnerScores.length,
        sequence: sequenceRef.current,
      });
      setPhase("settled");
      return;
    }

    // Nothing has a majority. Keep the previous settled sign on screen
    // rather than blanking it, but say the reading is unsettled.
    setPhase(tally.size > 0 ? "uncertain" : "watching");
  }, []);

  const runPrediction = useCallback(async () => {
    if (isPredictingRef.current) return;

    const snapshot = bufferRef.current.slice(-SEQUENCE_LENGTH);
    if (snapshot.length !== SEQUENCE_LENGTH) return;

    isPredictingRef.current = true;
    if (activeRef.current) setIsPredicting(true);

    try {
      const result = await predictKSL(snapshot);
      if (!activeRef.current) return;

      if (result.ok) {
        setLastPrediction(result.data);
        setLastError(null);
        integrate(result.data);
      } else {
        setLastError(result.error);
      }
    } catch (error) {
      if (activeRef.current) {
        setLastError(
          error instanceof Error
            ? error.message
            : "Recognition request failed."
        );
      }
    } finally {
      isPredictingRef.current = false;
      if (activeRef.current) setIsPredicting(false);
    }
  }, [integrate]);

  useEffect(() => {
    activeRef.current = true;
    let cancelled = false;
    let rafId: number | null = null;
    let holistic: HolisticInstance | null = null;

    bufferRef.current = [];
    historyRef.current = [];
    settledLabelRef.current = null;
    setFramesCollected(0);
    setLastPrediction(null);
    setStableSign(null);
    setLastError(null);

    if (!isCameraActive) {
      setPhase("idle");
      return () => {
        activeRef.current = false;
      };
    }

    setPhase("collecting");

    async function startRecognition() {
      try {
        const Holistic = await loadMediaPipeHolistic();
        if (cancelled) return;

        holistic = new Holistic({ locateFile: locateHolisticFile });

        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          refineFaceLandmarks: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        holistic.onResults((results) => {
          if (cancelled || !activeRef.current) return;

          const features = extractFeatures(results);
          if (!features) return;

          if (features.length !== FEATURE_LENGTH) {
            console.error(
              `[Fadhili AI] Invalid MediaPipe feature length: ${features.length}`
            );
            return;
          }

          const buffer = bufferRef.current;
          buffer.push(features);
          if (buffer.length > SEQUENCE_LENGTH) buffer.shift();
          setFramesCollected(buffer.length);

          if (buffer.length === SEQUENCE_LENGTH) {
            const now = performance.now();
            const enoughTimePassed =
              now - lastPredictTimeRef.current >= PREDICTION_INTERVAL_MS;

            if (!isPredictingRef.current && enoughTimePassed) {
              lastPredictTimeRef.current = now;
              void runPrediction();
            }
          }
        });

        async function processFrame() {
          if (cancelled || !holistic) return;

          const video = videoRef.current;
          if (
            video &&
            video.readyState >= 2 &&
            video.videoWidth > 0 &&
            video.videoHeight > 0
          ) {
            try {
              await holistic.send({ image: video });
            } catch (error) {
              console.error(
                "[Fadhili AI] MediaPipe frame processing failed:",
                error
              );
            }
          }

          if (!cancelled) {
            rafId = requestAnimationFrame(() => {
              void processFrame();
            });
          }
        }

        void processFrame();
      } catch (error) {
        if (cancelled || !activeRef.current) return;

        console.error(
          "[Fadhili AI] Failed to initialize MediaPipe Holistic:",
          error
        );
        setLastError(
          error instanceof Error
            ? error.message
            : "Failed to initialize MediaPipe Holistic."
        );
      }
    }

    void startRecognition();

    return () => {
      cancelled = true;
      activeRef.current = false;

      if (rafId !== null) cancelAnimationFrame(rafId);

      if (holistic) {
        try {
          const result = holistic.close();
          if (result instanceof Promise) {
            void result.catch((error) => {
              console.error("[Fadhili AI] Failed to close MediaPipe:", error);
            });
          }
        } catch (error) {
          console.error("[Fadhili AI] Failed to close MediaPipe:", error);
        }
      }

      bufferRef.current = [];
      historyRef.current = [];
      isPredictingRef.current = false;
    };
  }, [isCameraActive, videoRef, runPrediction]);

  return {
    framesCollected,
    isWarmingUp: framesCollected < SEQUENCE_LENGTH,
    isPredicting,
    phase,
    lastPrediction,
    stableSign,
    lastError,
    resetRecognition,
  };
}
