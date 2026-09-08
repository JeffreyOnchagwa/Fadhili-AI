/**
 * Fadhili AI — MediaPipe landmark capture.
 *
 * This file used to normalize landmarks in the browser and send
 * finished feature vectors to the API. It no longer does, deliberately.
 *
 * Why the change
 * --------------
 * Normalization existed in two places: here in TypeScript, and in the
 * Python training pipeline. The two were meant to match. They did — so
 * faithfully that when the Python side was found to divide every hand
 * coordinate by the *projected* wrist-to-middle-MCP distance (a value
 * that collapses under foreshortening and blew the features out of the
 * training distribution), the browser was reproducing exactly the same
 * bug, independently, with no way to notice.
 *
 * Two implementations of a preprocessing pipeline will drift, and when
 * they drift the model is silently fed something it was never trained
 * on. So the browser now captures RAW MediaPipe output and the server
 * builds features with the same module training uses. Parity is not a
 * matter of discipline any more; there is one implementation.
 *
 * Wire format — 150 raw values per frame:
 *   [0:24]    6 pose landmarks (MediaPipe 11..16) as x, y, z, visibility
 *   [24:87]   left hand,  21 landmarks as x, y, z
 *   [87:150]  right hand, 21 landmarks as x, y, z
 *
 * An absent hand is all zeros, matching MediaPipe's own encoding, which
 * is how the server detects a missing hand rather than one resting at
 * the origin.
 */

/** Frames sent per request. ~4 s of signing at the achievable rate. */
export const WINDOW_FRAMES = 48;

/** How much wall-clock time the buffer spans before being resampled. */
export const WINDOW_MS = 4000;

export const VALUES_PER_FRAME = 150;

/**
 * Pose landmarks the model uses: shoulders, elbows and wrists.
 * Must stay in this order — it is the order the server unpacks.
 */
const POSE_INDICES = [11, 12, 13, 14, 15, 16];

export type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility?: number;
};

export type MediaPipeResults = {
  poseLandmarks?: Landmark[] | null;
  leftHandLandmarks?: Landmark[] | null;
  rightHandLandmarks?: Landmark[] | null;
};

function finite(value: number | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/**
 * Extracts one frame of raw landmarks. Returns null when no pose was
 * detected at all, since a frame with no person carries nothing.
 */
export function extractRawLandmarks(
  results: MediaPipeResults
): number[] | null {
  const pose = results.poseLandmarks;
  if (!pose || pose.length <= 16) return null;

  const frame = new Array<number>(VALUES_PER_FRAME).fill(0);

  let offset = 0;
  for (const index of POSE_INDICES) {
    const landmark = pose[index];
    frame[offset++] = finite(landmark?.x);
    frame[offset++] = finite(landmark?.y);
    frame[offset++] = finite(landmark?.z);
    frame[offset++] = finite(landmark?.visibility);
  }

  // Hands stay at zero when MediaPipe reports none.
  const hands: [Landmark[] | null | undefined, number][] = [
    [results.leftHandLandmarks, 24],
    [results.rightHandLandmarks, 87],
  ];

  for (const [landmarks, base] of hands) {
    if (!landmarks || landmarks.length < 21) continue;
    for (let i = 0; i < 21; i++) {
      const landmark = landmarks[i];
      frame[base + i * 3] = finite(landmark?.x);
      frame[base + i * 3 + 1] = finite(landmark?.y);
      frame[base + i * 3 + 2] = finite(landmark?.z);
    }
  }

  return frame;
}

/**
 * Resamples a captured buffer to exactly WINDOW_FRAMES rows.
 *
 * The browser cannot hit a fixed frame rate — MediaPipe runs at
 * whatever the device manages — so the buffer is resampled by index to
 * a fixed length before being sent. Without this the server would
 * receive a different number of frames on every device and, worse, a
 * window covering a different amount of real time.
 */
export function resampleWindow(
  frames: number[][],
  target = WINDOW_FRAMES
): number[][] {
  if (frames.length === 0) return [];
  if (frames.length === target) return frames;

  const out: number[][] = [];
  const last = frames.length - 1;

  for (let i = 0; i < target; i++) {
    const position = (i / (target - 1)) * last;
    const low = Math.floor(position);
    const high = Math.min(low + 1, last);
    const weight = position - low;

    if (weight === 0) {
      out.push(frames[low]);
      continue;
    }

    const a = frames[low];
    const b = frames[high];
    const blended = new Array<number>(VALUES_PER_FRAME);
    for (let v = 0; v < VALUES_PER_FRAME; v++) {
      blended[v] = a[v] * (1 - weight) + b[v] * weight;
    }
    out.push(blended);
  }

  return out;
}

/** Rounds values before transport. 4 dp is ~0.2px on a 1920px frame. */
export function roundForTransport(frames: number[][]): number[][] {
  return frames.map((frame) => frame.map((v) => Math.round(v * 1e4) / 1e4));
}
