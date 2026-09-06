const HOLISTIC_VERSION = "0.5.1675471629";

const HOLISTIC_BASE_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@${HOLISTIC_VERSION}`;

const HOLISTIC_SCRIPT_URL =
  `${HOLISTIC_BASE_URL}/holistic.js`;

let loadingPromise: Promise<HolisticConstructor> | null = null;

export interface HolisticLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface HolisticResults {
  image: CanvasImageSource;
  poseLandmarks?: HolisticLandmark[];
  faceLandmarks?: HolisticLandmark[];
  leftHandLandmarks?: HolisticLandmark[];
  rightHandLandmarks?: HolisticLandmark[];
}

export interface HolisticOptions {
  modelComplexity?: 0 | 1 | 2;
  smoothLandmarks?: boolean;
  enableSegmentation?: boolean;
  smoothSegmentation?: boolean;
  refineFaceLandmarks?: boolean;
  minDetectionConfidence?: number;
  minTrackingConfidence?: number;
}

export interface HolisticInstance {
  setOptions(options: HolisticOptions): void;

  onResults(
    callback: (results: HolisticResults) => void
  ): void;

  send(input: {
    image: CanvasImageSource;
  }): Promise<void>;

  close(): Promise<void> | void;
}

export interface HolisticConstructor {
  new (config?: {
    locateFile?: (file: string) => string;
  }): HolisticInstance;
}

declare global {
  interface Window {
    Holistic?: HolisticConstructor;
  }
}

function getExistingConstructor():
  | HolisticConstructor
  | null {
  return typeof window !== "undefined" &&
    typeof window.Holistic === "function"
    ? window.Holistic
    : null;
}

export function loadMediaPipeHolistic():
  Promise<HolisticConstructor> {
  const existing = getExistingConstructor();

  if (existing) {
    return Promise.resolve(existing);
  }

  if (loadingPromise) {
    return loadingPromise;
  }

  loadingPromise = new Promise<HolisticConstructor>(
    (resolve, reject) => {
      if (typeof document === "undefined") {
        loadingPromise = null;

        reject(
          new Error(
            "MediaPipe Holistic can only run in a browser."
          )
        );

        return;
      }

      const existingScript =
        document.querySelector<HTMLScriptElement>(
          `script[data-fadhili-mediapipe-holistic="true"]`
        );

      const resolveConstructor = () => {
        const Constructor = getExistingConstructor();

        if (!Constructor) {
          loadingPromise = null;

          reject(
            new Error(
              "MediaPipe Holistic loaded, but window.Holistic was not created."
            )
          );

          return;
        }

        resolve(Constructor);
      };

      if (existingScript) {
        if (getExistingConstructor()) {
          resolveConstructor();
          return;
        }

        existingScript.addEventListener(
          "load",
          resolveConstructor,
          { once: true }
        );

        existingScript.addEventListener(
          "error",
          () => {
            loadingPromise = null;

            reject(
              new Error(
                "Failed to load MediaPipe Holistic."
              )
            );
          },
          { once: true }
        );

        return;
      }

      const script = document.createElement("script");

      script.src = HOLISTIC_SCRIPT_URL;
      script.async = true;
      script.crossOrigin = "anonymous";

      script.dataset.fadhiliMediapipeHolistic = "true";

      script.addEventListener(
        "load",
        resolveConstructor,
        { once: true }
      );

      script.addEventListener(
        "error",
        () => {
          loadingPromise = null;

          script.remove();

          reject(
            new Error(
              "Failed to download MediaPipe Holistic."
            )
          );
        },
        { once: true }
      );

      document.head.appendChild(script);
    }
  );

  return loadingPromise;
}

export function locateHolisticFile(
  file: string
): string {
  return `${HOLISTIC_BASE_URL}/${file}`;
}