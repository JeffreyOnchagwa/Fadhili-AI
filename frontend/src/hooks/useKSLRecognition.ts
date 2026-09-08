import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

import { predictKSLv2 } from "../services/api";
import type { KSLPredictionV2Response } from "../services/api";
import {
  extractRawLandmarks,
  resampleWindow,
  roundForTransport,
  WINDOW_FRAMES,
  WINDOW_MS,
} from "../services/mediapipeFeatures";
import {
  loadMediaPipeHolistic,
  locateHolisticFile,
} from "../services/mediapipeHolisticLoader";
import type { HolisticInstance } from "../services/mediapipeHolisticLoader";

const PREDICTION_INTERVAL_MS = 1000;

/** A point in MediaPipe's normalized [0,1] image space. */
export interface LandmarkPoint {
  x: number;
  y: number;
}

/**
 * The latest frame's landmark positions, for the skeleton overlay.
 *
 * Populated on every MediaPipe callback regardless of the prediction
 * cadence, so the overlay can redraw every frame. This is read from a
 * ref rather than React state deliberately: at up to 30fps, routing it
 * through setState would force a re-render on every single video frame.
 */
export interface LiveLandmarks {
  pose: LandmarkPoint[] | null;
  leftHand: LandmarkPoint[] | null;
  rightHand: LandmarkPoint[] | null;
}

/**
 * Temporal stabilization.
 *
 * The model is overconfident — on held-out signers it averaged 0.92
 * confidence on predictions that were WRONG — so confidence alone is a
 * poor gate, and surfacing every raw inference makes the panel flicker
 * between unrelated words. A sign is only presented once it wins a
 * majority of a short rolling window. This cannot make the model
 * correct; it stops the UI presenting noise as a reading.
 */
const HISTORY_WINDOW = 5;
const AGREEMENT_REQUIRED = 3;

export type RecognitionPhase =
  | "idle"
  | "collecting"
  | "watching"
  | "uncertain"
  | "no_sign"
  | "settled";

export interface StableSign {
  label: string;
  meanConfidence: number;
  agreement: number;
  /** Increments only when a NEW sign settles; drives speech. */
  sequence: number;
}

export interface UseKSLRecognitionResult {
  framesCollected: number;
  isWarmingUp: boolean;
  isPredicting: boolean;
  phase: RecognitionPhase;
  lastPrediction: KSLPredictionV2Response | null;
  stableSign: StableSign | null;
  lastError: string | null;
  resetRecognition: () => void;
  /** Latest landmark positions, for a skeleton overlay. See LiveLandmarks. */
  landmarksRef: RefObject<LiveLandmarks | null>;
}

interface BufferedFrame {
  values: number[];
  at: number;
}

export function useKSLRecognition(
  videoRef: RefObject<HTMLVideoElement | null>,
  isCameraActive: boolean
): UseKSLRecognitionResult {
  const [framesCollected, setFramesCollected] = useState(0);
  const [isPredicting, setIsPredicting] = useState(false);
  const [lastPrediction, setLastPrediction] =
    useState<KSLPredictionV2Response | null>(null);
  const [stableSign, setStableSign] = useState<StableSign | null>(null);
  const [phase, setPhase] = useState<RecognitionPhase>("idle");
  const [lastError, setLastError] = useState<string | null>(null);

  const bufferRef = useRef<BufferedFrame[]>([]);
  const landmarksRef = useRef<LiveLandmarks | null>(null);
  const historyRef = useRef<(string | null)[]>([]);
  const confidenceRef = useRef<Map<string, number[]>>(new Map());
  const settledLabelRef = useRef<string | null>(null);
  const sequenceRef = useRef(0);
  const isPredictingRef = useRef(false);
  const lastPredictTimeRef = useRef(0);
  const activeRef = useRef(false);

  const resetRecognition = useCallback(() => {
    bufferRef.current = [];
    historyRef.current = [];
    confidenceRef.current = new Map();
    settledLabelRef.current = null;
    lastPredictTimeRef.current = 0;
    setFramesCollected(0);
    setLastPrediction(null);
    setStableSign(null);
    setLastError(null);
    setPhase(activeRef.current ? "collecting" : "idle");
  }, []);

  /** Folds one inference into the rolling window. */
  const integrate = useCallback((response: KSLPredictionV2Response) => {
    // The server's motion gate is authoritative about "nobody is
    // signing". Clear the settled sign rather than leaving a stale word
    // on screen after the user has stopped.
    if (response.reason === "no_sign_detected") {
      historyRef.current = [];
      confidenceRef.current = new Map();
      settledLabelRef.current = null;
      setStableSign(null);
      setPhase("no_sign");
      return;
    }

    const history = historyRef.current;
    const label = response.accepted ? response.prediction : null;
    history.push(label);
    if (history.length > HISTORY_WINDOW) history.shift();

    const counts = new Map<string, number>();
    for (const entry of history) {
      if (!entry) continue;
      counts.set(entry, (counts.get(entry) ?? 0) + 1);
    }
    if (label) {
      const scores = confidenceRef.current.get(label) ?? [];
      scores.push(response.confidence);
      confidenceRef.current.set(label, scores.slice(-HISTORY_WINDOW));
    }

    let winner: string | null = null;
    let winnerCount = 0;
    for (const [entry, count] of counts) {
      if (count > winnerCount) {
        winner = entry;
        winnerCount = count;
      }
    }

    if (winner && winnerCount >= AGREEMENT_REQUIRED) {
      const scores = confidenceRef.current.get(winner) ?? [
        response.confidence,
      ];
      const meanConfidence =
        scores.reduce((sum, value) => sum + value, 0) / scores.length;

      if (settledLabelRef.current !== winner) {
        settledLabelRef.current = winner;
        sequenceRef.current += 1;
      }

      setStableSign({
        label: winner,
        meanConfidence,
        agreement: winnerCount,
        sequence: sequenceRef.current,
      });
      setPhase("settled");
      return;
    }

    setPhase(counts.size > 0 ? "uncertain" : "watching");
  }, []);

  const runPrediction = useCallback(async () => {
    if (isPredictingRef.current) return;

    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    const cutoff = performance.now() - WINDOW_MS;
    const recent = bufferRef.current.filter((frame) => frame.at >= cutoff);
    if (recent.length < 16) return;

    isPredictingRef.current = true;
    if (activeRef.current) setIsPredicting(true);

    try {
      const window = roundForTransport(
        resampleWindow(
          recent.map((frame) => frame.values),
          WINDOW_FRAMES
        )
      );

      const result = await predictKSLv2(
        window,
        video.videoWidth,
        video.videoHeight
      );
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
  }, [integrate, videoRef]);

  useEffect(() => {
    activeRef.current = true;
    let cancelled = false;
    let rafId: number | null = null;
    let holistic: HolisticInstance | null = null;

    bufferRef.current = [];
    historyRef.current = [];
    confidenceRef.current = new Map();
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

          // Drawing data for the skeleton overlay. Written every frame
          // regardless of prediction cadence; read straight from the
          // ref by a canvas draw loop rather than through React state,
          // so a 30fps camera does not force 30 re-renders a second.
          landmarksRef.current = {
            pose: results.poseLandmarks
              ? results.poseLandmarks.map((p) => ({ x: p.x, y: p.y }))
              : null,
            leftHand: results.leftHandLandmarks
              ? results.leftHandLandmarks.map((p) => ({ x: p.x, y: p.y }))
              : null,
            rightHand: results.rightHandLandmarks
              ? results.rightHandLandmarks.map((p) => ({ x: p.x, y: p.y }))
              : null,
          };

          const frame = extractRawLandmarks(results);
          if (!frame) return;

          const now = performance.now();
          const buffer = bufferRef.current;
          buffer.push({ values: frame, at: now });

          // Keep a little more than one window so a slow device still
          // has a full window's worth of history to resample from.
          const cutoff = now - WINDOW_MS * 1.5;
          while (buffer.length && buffer[0].at < cutoff) buffer.shift();

          setFramesCollected(
            buffer.filter((f) => f.at >= now - WINDOW_MS).length
          );

          if (now - lastPredictTimeRef.current >= PREDICTION_INTERVAL_MS) {
            lastPredictTimeRef.current = now;
            void runPrediction();
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
      landmarksRef.current = null;
      isPredictingRef.current = false;
    };
  }, [isCameraActive, videoRef, runPrediction]);

  return {
    framesCollected,
    landmarksRef,
    // Warming up until the buffer covers enough of the window to be
    // worth sending.
    isWarmingUp: framesCollected < 16,
    isPredicting,
    phase,
    lastPrediction,
    stableSign,
    lastError,
    resetRecognition,
  };
}
