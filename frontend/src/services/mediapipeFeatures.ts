/**
 * Converts one browser-side MediaPipe Holistic result into the exact
 * 1662-length feature vector the KSL model expects, in the same
 * pose+face+left-hand+right-hand order used by the Python training
 * pipeline. Kept dependency-free (no import of @mediapipe/holistic
 * types) so it's easy to unit-test and reuse.
 */

export const POSE_LANDMARK_COUNT = 33;
export const FACE_LANDMARK_COUNT = 468;
export const HAND_LANDMARK_COUNT = 21;

const POSE_FEATURE_LENGTH = POSE_LANDMARK_COUNT * 4; // x, y, z, visibility
const FACE_FEATURE_LENGTH = FACE_LANDMARK_COUNT * 3; // x, y, z
const HAND_FEATURE_LENGTH = HAND_LANDMARK_COUNT * 3; // x, y, z

export const FEATURE_LENGTH =
  POSE_FEATURE_LENGTH + FACE_FEATURE_LENGTH + HAND_FEATURE_LENGTH + HAND_FEATURE_LENGTH; // 1662
export const SEQUENCE_LENGTH = 30;

export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/** Minimal shape we need from a MediaPipe Holistic result. */
export interface HolisticResultsLike {
  poseLandmarks?: Landmark[];
  faceLandmarks?: Landmark[];
  leftHandLandmarks?: Landmark[];
  rightHandLandmarks?: Landmark[];
}

function flattenPose(landmarks: Landmark[] | undefined): number[] {
  if (!landmarks || landmarks.length !== POSE_LANDMARK_COUNT) {
    return new Array(POSE_FEATURE_LENGTH).fill(0);
  }
  const out: number[] = [];
  for (const lm of landmarks) {
    out.push(lm.x, lm.y, lm.z, lm.visibility ?? 0);
  }
  return out;
}

function flattenXYZ(landmarks: Landmark[] | undefined, expectedCount: number): number[] {
  const expectedLength = expectedCount * 3;
  if (!landmarks || landmarks.length !== expectedCount) {
    return new Array(expectedLength).fill(0);
  }
  const out: number[] = [];
  for (const lm of landmarks) {
    out.push(lm.x, lm.y, lm.z);
  }
  return out;
}

/**
 * Flattens one Holistic result into a 1662-length feature array, or
 * returns null if something is malformed (e.g. an unexpected landmark
 * count). Any landmark group MediaPipe didn't detect this frame is
 * zero-filled at its expected length rather than omitted, so the
 * total length is always exactly 1662 when a value is returned.
 */
export function extractFeatures(results: HolisticResultsLike): number[] | null {
  const pose = flattenPose(results.poseLandmarks);
  const face = flattenXYZ(results.faceLandmarks, FACE_LANDMARK_COUNT);
  const leftHand = flattenXYZ(results.leftHandLandmarks, HAND_LANDMARK_COUNT);
  const rightHand = flattenXYZ(results.rightHandLandmarks, HAND_LANDMARK_COUNT);

  const features = [...pose, ...face, ...leftHand, ...rightHand];

  if (features.length !== FEATURE_LENGTH) {
    // Defensive guard — should be unreachable given the checks above,
    // but a corrupt frame must never silently reach the network.
    console.error(
      `[Fadhili AI] Extracted ${features.length} features, expected ${FEATURE_LENGTH}. Dropping frame.`
    );
    return null;
  }
  return features;
}