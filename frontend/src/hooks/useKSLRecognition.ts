import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import type { RefObject } from "react";

import type { SupportedSignLanguage } from "../types";

import {
  predictKSL,
} from "../services/api";

import type {
  KSLPredictionResponse,
} from "../services/api";

import {
  extractFeatures,
  SEQUENCE_LENGTH,
} from "../services/mediapipeFeatures";

import {
  loadMediaPipeHolistic,
  locateHolisticFile,
} from "../services/mediapipeHolisticLoader";

import type {
  HolisticInstance,
} from "../services/mediapipeHolisticLoader";

const PREDICTION_INTERVAL_MS = 800;

export interface UseKSLRecognitionResult {
  isSupportedLanguage: boolean;

  comingSoonMessage: string | null;

  framesCollected: number;

  isWarmingUp: boolean;

  isPredicting: boolean;

  lastPrediction:
    | KSLPredictionResponse
    | null;

  lastError: string | null;

  resetRecognition: () => void;
}

export function useKSLRecognition(
  videoRef: RefObject<HTMLVideoElement | null>,
  language: SupportedSignLanguage,
  isCameraActive: boolean
): UseKSLRecognitionResult {
  const [
    framesCollected,
    setFramesCollected,
  ] = useState(0);

  const [
    isPredicting,
    setIsPredicting,
  ] = useState(false);

  const [
    lastPrediction,
    setLastPrediction,
  ] =
    useState<KSLPredictionResponse | null>(
      null
    );

  const [
    lastError,
    setLastError,
  ] = useState<string | null>(null);

  const bufferRef =
    useRef<number[][]>([]);

  const isPredictingRef =
    useRef(false);

  const lastPredictTimeRef =
    useRef(0);

  const activeRef =
    useRef(false);

  const isSupportedLanguage =
    language === "KSL";

  const comingSoonMessage =
    isSupportedLanguage
      ? null
      : `${language} recognition coming soon`;

  const resetRecognition =
    useCallback(() => {
      bufferRef.current = [];

      lastPredictTimeRef.current = 0;

      setFramesCollected(0);

      setLastPrediction(null);

      setLastError(null);
    }, []);

  const runPrediction =
    useCallback(async () => {
      if (isPredictingRef.current) {
        return;
      }

      const snapshot =
        bufferRef.current.slice(
          -SEQUENCE_LENGTH
        );

      if (
        snapshot.length !==
        SEQUENCE_LENGTH
      ) {
        return;
      }

      isPredictingRef.current = true;

      if (activeRef.current) {
        setIsPredicting(true);
      }

      try {
        const result =
          await predictKSL(snapshot);

        if (!activeRef.current) {
          return;
        }

        if (result.ok === true) {
          setLastPrediction(
            result.data
          );

          setLastError(null);
        } else {
          setLastError(
            result.error
          );
        }
      } catch (error) {
        if (
          activeRef.current
        ) {
          setLastError(
            error instanceof Error
              ? error.message
              : "Recognition request failed."
          );
        }
      } finally {
        isPredictingRef.current =
          false;

        if (activeRef.current) {
          setIsPredicting(false);
        }
      }
    }, []);

  useEffect(() => {
    activeRef.current = true;

    let cancelled = false;

    let rafId:
      | number
      | null = null;

    let holistic:
      | HolisticInstance
      | null = null;

    if (
      !isCameraActive ||
      language !== "KSL"
    ) {
      bufferRef.current = [];

      setFramesCollected(0);

      setLastPrediction(null);

      setLastError(null);

      return () => {
        activeRef.current =
          false;
      };
    }

    bufferRef.current = [];

    setFramesCollected(0);

    setLastPrediction(null);

    setLastError(null);

    async function startRecognition() {
      try {
        const Holistic =
          await loadMediaPipeHolistic();

        if (cancelled) {
          return;
        }

        holistic =
          new Holistic({
            locateFile:
              locateHolisticFile,
          });

        holistic.setOptions({
          modelComplexity: 1,

          smoothLandmarks: true,

          enableSegmentation: false,

          smoothSegmentation: false,

          // Critical:
          // Python training expected
          // exactly 468 face landmarks.
          refineFaceLandmarks: false,

          minDetectionConfidence:
            0.5,

          minTrackingConfidence:
            0.5,
        });

        holistic.onResults(
          (results) => {
            if (
              cancelled ||
              !activeRef.current
            ) {
              return;
            }

            const features =
              extractFeatures(
                results
              );

            if (!features) {
              return;
            }

            if (
              features.length !==
              150
            ) {
              console.error(
                `[Fadhili AI] Invalid MediaPipe feature length: ${features.length}`
              );

              return;
            }

            const buffer =
              bufferRef.current;

            buffer.push(
              features
            );

            if (
              buffer.length >
              SEQUENCE_LENGTH
            ) {
              buffer.shift();
            }

            setFramesCollected(
              buffer.length
            );

            if (
              buffer.length ===
              SEQUENCE_LENGTH
            ) {
              const now =
                performance.now();

              const enoughTimePassed =
                now -
                  lastPredictTimeRef.current >=
                PREDICTION_INTERVAL_MS;

              if (
                !isPredictingRef.current &&
                enoughTimePassed
              ) {
                lastPredictTimeRef.current =
                  now;

                void runPrediction();
              }
            }
          }
        );

        async function processFrame() {
          if (
            cancelled ||
            !holistic
          ) {
            return;
          }

          const video =
            videoRef.current;

          if (
            video &&
            video.readyState >= 2 &&
            video.videoWidth > 0 &&
            video.videoHeight > 0
          ) {
            try {
              await holistic.send({
                image: video,
              });
            } catch (error) {
              console.error(
                "[Fadhili AI] MediaPipe frame processing failed:",
                error
              );
            }
          }

          if (!cancelled) {
            rafId =
              requestAnimationFrame(
                () => {
                  void processFrame();
                }
              );
          }
        }

        void processFrame();
      } catch (error) {
        if (
          cancelled ||
          !activeRef.current
        ) {
          return;
        }

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

      activeRef.current =
        false;

      if (rafId !== null) {
        cancelAnimationFrame(
          rafId
        );
      }

      if (holistic) {
        try {
          const result =
            holistic.close();

          if (
            result instanceof
            Promise
          ) {
            void result.catch(
              (error) => {
                console.error(
                  "[Fadhili AI] Failed to close MediaPipe:",
                  error
                );
              }
            );
          }
        } catch (error) {
          console.error(
            "[Fadhili AI] Failed to close MediaPipe:",
            error
          );
        }
      }

      bufferRef.current = [];

      isPredictingRef.current =
        false;
    };
  }, [
    isCameraActive,
    language,
    videoRef,
    runPrediction,
  ]);

  return {
    isSupportedLanguage,

    comingSoonMessage,

    framesCollected,

    isWarmingUp:
      framesCollected <
      SEQUENCE_LENGTH,

    isPredicting,

    lastPrediction,

    lastError,

    resetRecognition,
  };
}
