"""
Fadhili AI - Kenyan Sign Language Webcam Recognition Test

Captures webcam frames, extracts MediaPipe Holistic landmarks,
builds a 30-frame sequence, and runs the pretrained KSL model.
"""

import os
import sys
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import keras


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join("ksl-model-source", "best_model.h5")

SEQUENCE_LENGTH = 30
FEATURE_LENGTH = 1662

POSE_LANDMARKS = 33
POSE_VALUES_PER_LANDMARK = 4

FACE_LANDMARKS = 468
FACE_VALUES_PER_LANDMARK = 3

HAND_LANDMARKS = 21
HAND_VALUES_PER_LANDMARK = 3

POSE_FEATURE_LENGTH = POSE_LANDMARKS * POSE_VALUES_PER_LANDMARK
FACE_FEATURE_LENGTH = FACE_LANDMARKS * FACE_VALUES_PER_LANDMARK
HAND_FEATURE_LENGTH = HAND_LANDMARKS * HAND_VALUES_PER_LANDMARK

assert (
    POSE_FEATURE_LENGTH
    + FACE_FEATURE_LENGTH
    + HAND_FEATURE_LENGTH
    + HAND_FEATURE_LENGTH
    == FEATURE_LENGTH
), "Feature length must equal 1662."


# ============================================================
# MODEL LABELS
# IMPORTANT: DO NOT REORDER
# ============================================================

CLASS_NAMES = [
    "me",
    "you",
    "friend",
    "name",
    "mine",
    "who",
    "how",
    "please",
    "help-me",
    "wait",
    "now",
    "home",
    "where",
    "give-me",
    "thank-you",
    "polite",
    "hello",
    "good",
    "mother",
    "father",
    "uncle",
    "cousing",
    "brother",
    "sister",
    "doughter",
    "parent",
    "relative",
    "yes",
    "no",
    "sorry",
]

DISPLAY_NAME_OVERRIDES = {
    "cousing": "cousin",
    "doughter": "daughter",
}

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 5


def to_display_name(class_name: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(class_name, class_name)


# ============================================================
# MEDIAPIPE FEATURE EXTRACTION
# ============================================================

def extract_keypoints(results) -> np.ndarray:
    """
    Convert MediaPipe Holistic landmarks into exactly 1662 features.

    Order:
        Pose       = 33 * 4   = 132
        Face       = 468 * 3  = 1404
        Left hand  = 21 * 3   = 63
        Right hand = 21 * 3   = 63

        Total = 1662
    """

    if results.pose_landmarks:
        pose = np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(POSE_FEATURE_LENGTH, dtype=np.float32)

    if results.face_landmarks:
        face = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.face_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
    else:
        face = np.zeros(FACE_FEATURE_LENGTH, dtype=np.float32)

    if results.left_hand_landmarks:
        left_hand = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
    else:
        left_hand = np.zeros(HAND_FEATURE_LENGTH, dtype=np.float32)

    if results.right_hand_landmarks:
        right_hand = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
    else:
        right_hand = np.zeros(HAND_FEATURE_LENGTH, dtype=np.float32)

    keypoints = np.concatenate(
        [
            pose,
            face,
            left_hand,
            right_hand,
        ]
    ).astype(np.float32)

    if keypoints.shape[0] != FEATURE_LENGTH:
        raise ValueError(
            f"Expected {FEATURE_LENGTH} features, "
            f"but extracted {keypoints.shape[0]}"
        )

    return keypoints


# ============================================================
# LANDMARK VISUALIZATION
# ============================================================

def draw_landmarks(image, results, mp_holistic, mp_drawing):
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
        )

    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )


# ============================================================
# MODEL LOADING
# ============================================================

def load_ksl_model():
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print(
            "[ERROR] Run this script from the backend directory "
            "and make sure ksl-model-source/best_model.h5 exists."
        )
        sys.exit(1)

    print(f"[INFO] Loading KSL model from: {MODEL_PATH}")

    try:
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )
    except Exception as exc:
        print(f"[ERROR] Could not load KSL model: {exc}")
        sys.exit(1)

    print("[INFO] Model loaded successfully.")
    print(f"[INFO] Input shape:  {model.input_shape}")
    print(f"[INFO] Output shape: {model.output_shape}")

    expected_input = (None, SEQUENCE_LENGTH, FEATURE_LENGTH)

    if tuple(model.input_shape) != expected_input:
        print(
            f"[WARNING] Expected model input {expected_input}, "
            f"but model reports {model.input_shape}"
        )

    if model.output_shape[-1] != len(CLASS_NAMES):
        print(
            "[ERROR] Model output count does not match "
            "the number of class labels."
        )
        sys.exit(1)

    return model


# ============================================================
# MAIN WEBCAM LOOP
# ============================================================

def main():
    model = load_ksl_model()

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    print("[INFO] Opening webcam...")

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print(
            "[ERROR] Check Windows camera permissions and make sure "
            "another application is not using the camera."
        )
        return

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    recent_votes = deque(maxlen=SMOOTHING_WINDOW)

    raw_class_name = None
    raw_confidence = 0.0

    stable_class_name = None
    last_announced_class = None

    print("[INFO] Webcam opened successfully.")
    print("[INFO] Starting KSL recognition.")
    print("[INFO] Press Q in the webcam window to quit.")

    try:
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:

            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    print("[ERROR] Failed to read webcam frame.")
                    break

                # Mirror view
            

                # OpenCV BGR -> MediaPipe RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                rgb_frame.flags.writeable = False

                results = holistic.process(rgb_frame)

                rgb_frame.flags.writeable = True

                # Draw pose and hand landmarks
                draw_landmarks(
                    frame,
                    results,
                    mp_holistic,
                    mp_drawing,
                )

                # Extract exactly 1662 features
                keypoints = extract_keypoints(results)

                sequence.append(keypoints)

                # ====================================================
                # MODEL INFERENCE
                # ====================================================

                if len(sequence) == SEQUENCE_LENGTH:

                    input_data = np.array(
                        sequence,
                        dtype=np.float32,
                    )

                    input_data = np.expand_dims(
                        input_data,
                        axis=0,
                    )

                    if input_data.shape != (
                        1,
                        SEQUENCE_LENGTH,
                        FEATURE_LENGTH,
                    ):
                        raise ValueError(
                            f"Invalid model input shape: "
                            f"{input_data.shape}"
                        )

                    probabilities = model.predict(
                        input_data,
                        verbose=0,
                    )[0]

                    predicted_idx = int(
                        np.argmax(probabilities)
                    )

                    raw_confidence = float(
                        probabilities[predicted_idx]
                    )

                    raw_class_name = CLASS_NAMES[
                        predicted_idx
                    ]

                    # ================================================
                    # PREDICTION STABILIZATION
                    # ================================================

                    if raw_confidence >= CONFIDENCE_THRESHOLD:
                        recent_votes.append(raw_class_name)
                    else:
                        recent_votes.append(None)

                    if (
                        len(recent_votes) == SMOOTHING_WINDOW
                        and recent_votes[0] is not None
                        and all(
                            vote == recent_votes[0]
                            for vote in recent_votes
                        )
                    ):
                        confirmed_class = recent_votes[0]

                        stable_class_name = confirmed_class

                        if confirmed_class != last_announced_class:

                            display_sign = to_display_name(
                                confirmed_class
                            )

                            print(
                                f"[STABLE] {display_sign} "
                                f"({raw_confidence * 100:.1f}%)"
                            )

                            last_announced_class = confirmed_class

                # ====================================================
                # DISPLAY
                # ====================================================

                if len(sequence) < SEQUENCE_LENGTH:

                    cv2.putText(
                        frame,
                        f"Warming up: "
                        f"{len(sequence)}/{SEQUENCE_LENGTH} frames",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                else:

                    raw_display = (
                        to_display_name(raw_class_name)
                        if raw_class_name
                        else "-"
                    )

                    stable_display = (
                        to_display_name(stable_class_name)
                        if stable_class_name
                        else "-"
                    )

                    cv2.putText(
                        frame,
                        f"Raw: {raw_display} "
                        f"({raw_confidence * 100:.1f}%)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        frame,
                        f"Stable: {stable_display}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "Fadhili AI - KSL Recognition",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Q pressed. Stopping...")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam released.")
        print("[INFO] Fadhili AI webcam test stopped.")


if __name__ == "__main__":
    main()