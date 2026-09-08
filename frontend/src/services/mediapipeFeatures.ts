/**
 * Fadhili AI
 * MediaPipe feature extraction for the KSL recognition model.
 *
 * Model input:
 *   30 frames × 150 features
 *
 * Per-frame feature layout:
 *   Pose:       6 landmarks × 4 values = 24
 *   Left hand: 21 landmarks × 3 values = 63
 *   Right hand:21 landmarks × 3 values = 63
 *
 * Total:
 *   24 + 63 + 63 = 150
 */

export const SEQUENCE_LENGTH = 30;
export const FEATURES_PER_FRAME = 150;

/**
 * Pose landmarks used by the Python training pipeline:
 *
 * 11 = left shoulder
 * 12 = right shoulder
 * 13 = left elbow
 * 14 = right elbow
 * 15 = left wrist
 * 16 = right wrist
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

/**
 * Creates a zero-filled feature vector.
 */
function zeros(length: number): number[] {
  return new Array<number>(length).fill(0);
}

/**
 * Normalizes one hand.
 *
 * This mirrors the Python training preprocessing:
 *
 * 1. Landmark 0 (wrist) is the origin.
 * 2. Wrist -> middle-finger MCP (landmark 9) is the scale.
 * 3. Every landmark is represented relative to the wrist.
 *
 * Output:
 *   21 landmarks × XYZ = 63 features
 */
function normalizeHand(
  landmarks?: Landmark[] | null
): number[] {
  if (!landmarks || landmarks.length < 21) {
    return zeros(63);
  }

  const wrist = landmarks[0];
  const middleFingerMcp = landmarks[9];

  if (!wrist || !middleFingerMcp) {
    return zeros(63);
  }

  const dx = middleFingerMcp.x - wrist.x;
  const dy = middleFingerMcp.y - wrist.y;
  const dz = middleFingerMcp.z - wrist.z;

  const distance = Math.sqrt(
    dx * dx +
    dy * dy +
    dz * dz
  );

  const scale =
    Number.isFinite(distance) && distance > 1e-6
      ? distance
      : 1;

  const features: number[] = [];

  for (let i = 0; i < 21; i++) {
    const landmark = landmarks[i];

    if (!landmark) {
      features.push(0, 0, 0);
      continue;
    }

    const x = (landmark.x - wrist.x) / scale;
    const y = (landmark.y - wrist.y) / scale;
    const z = (landmark.z - wrist.z) / scale;

    features.push(
      Number.isFinite(x) ? x : 0,
      Number.isFinite(y) ? y : 0,
      Number.isFinite(z) ? z : 0
    );
  }

  if (features.length !== 63) {
    throw new Error(
      `Hand feature extraction failed: expected 63 features, got ${features.length}`
    );
  }

  return features;
}

/**
 * Normalizes the upper-body pose.
 *
 * This mirrors the Python training preprocessing:
 *
 * 1. Midpoint between shoulders becomes the origin.
 * 2. Shoulder-to-shoulder distance becomes the scale.
 * 3. Six upper-body landmarks are retained.
 * 4. Each contributes X, Y, Z and visibility.
 *
 * Output:
 *   6 landmarks × 4 = 24 features
 */
function normalizePose(
  landmarks?: Landmark[] | null
): number[] {
  if (!landmarks || landmarks.length <= 16) {
    return zeros(24);
  }

  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];

  if (!leftShoulder || !rightShoulder) {
    return zeros(24);
  }

  const centerX =
    (leftShoulder.x + rightShoulder.x) / 2;

  const centerY =
    (leftShoulder.y + rightShoulder.y) / 2;

  const centerZ =
    (leftShoulder.z + rightShoulder.z) / 2;

  const dx =
    leftShoulder.x - rightShoulder.x;

  const dy =
    leftShoulder.y - rightShoulder.y;

  const dz =
    leftShoulder.z - rightShoulder.z;

  const shoulderDistance = Math.sqrt(
    dx * dx +
    dy * dy +
    dz * dz
  );

  const scale =
    Number.isFinite(shoulderDistance) &&
    shoulderDistance > 1e-6
      ? shoulderDistance
      : 1;

  const features: number[] = [];

  for (const index of POSE_INDICES) {
    const landmark = landmarks[index];

    if (!landmark) {
      features.push(0, 0, 0, 0);
      continue;
    }

    const x =
      (landmark.x - centerX) / scale;

    const y =
      (landmark.y - centerY) / scale;

    const z =
      (landmark.z - centerZ) / scale;

    const visibility =
      landmark.visibility ?? 0;

    features.push(
      Number.isFinite(x) ? x : 0,
      Number.isFinite(y) ? y : 0,
      Number.isFinite(z) ? z : 0,
      Number.isFinite(visibility)
        ? visibility
        : 0
    );
  }

  if (features.length !== 24) {
    throw new Error(
      `Pose feature extraction failed: expected 24 features, got ${features.length}`
    );
  }

  return features;
}

/**
 * Extracts the complete 150-feature vector from one
 * MediaPipe Holistic result.
 *
 * Feature order MUST remain:
 *
 *   pose
 *   left hand
 *   right hand
 *
 * because this is the order used during model training.
 */
export function extractMediaPipeFeatures(
  results: MediaPipeResults
): number[] {
  const poseFeatures =
    normalizePose(results.poseLandmarks);

  const leftHandFeatures =
    normalizeHand(results.leftHandLandmarks);

  const rightHandFeatures =
    normalizeHand(results.rightHandLandmarks);

  const features = [
    ...poseFeatures,
    ...leftHandFeatures,
    ...rightHandFeatures,
  ];

  if (features.length !== FEATURES_PER_FRAME) {
    throw new Error(
      `MediaPipe feature extraction failed: expected ${FEATURES_PER_FRAME} features, got ${features.length}`
    );
  }

  for (let i = 0; i < features.length; i++) {
    if (!Number.isFinite(features[i])) {
      features[i] = 0;
    }
  }

  return features;
}

/**
 * Backwards-compatible export.
 *
 * useKSLRecognition.ts already imports:
 *
 *   extractFeatures
 *
 * so we preserve that API instead of forcing changes
 * throughout the frontend.
 */
export const extractFeatures =
  extractMediaPipeFeatures;