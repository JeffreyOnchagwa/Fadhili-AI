"""
Fadhili AI
KSL V4 Motion-Aware MediaPipe Feature Extraction

Pipeline
--------
Original video
    ->
MediaPipe Holistic on every decoded frame
    ->
150-dimensional normalized landmark vector per frame
    ->
landmark-motion analysis
    ->
active signing interval detection
    ->
context padding
    ->
temporal normalization to 30 frames
    ->
(30, 150) NumPy feature sequence

V4 does NOT overwrite the older feature dataset.

Output:
    backend/training/data/features_v4/

Metadata:
    backend/training/data/metadata/fadhili_v4_features.json
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import json
import sys

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VIDEO_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "videos"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "features_v4"
)

METADATA_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "metadata"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 150

SPLITS = [
    "train",
    "val",
    "test",
]

CLASSES = [
    "Agreement",
    "Friend",
    "Gift",
    "Market",
    "Monday",
    "Picture",
    "Proud",
    "Teach",
    "Twin",
    "Ugali",
]

EXPECTED_VIDEO_COUNT = 742


# ============================================================
# V4 MOTION CONFIGURATION
# ============================================================

MOTION_THRESHOLD_RATIO = 0.25

MOTION_MIN_THRESHOLD = 0.015

# Amount of temporal context added around the detected sign.
MOTION_CONTEXT_RATIO = 0.08

# Never allow the raw active interval to become extremely short.
MIN_ACTIVE_RAW_FRAMES = 8


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


# ============================================================
# POSE LANDMARKS
# ============================================================
#
# MediaPipe Pose:
#
# 11 = left shoulder
# 12 = right shoulder
# 13 = left elbow
# 14 = right elbow
# 15 = left wrist
# 16 = right wrist
#
# Each pose landmark contributes:
#
# x, y, z, visibility
#
# 6 * 4 = 24
# ============================================================

POSE_INDICES = [
    11,
    12,
    13,
    14,
    15,
    16,
]


# ============================================================
# EXTRACT RAW POSE
# ============================================================

def extract_pose(results):
    """
    Extract six upper-body pose landmarks.

    Output:
        24 values

    Layout per landmark:
        x, y, z, visibility
    """

    if results.pose_landmarks is None:

        return np.zeros(
            24,
            dtype=np.float32,
        )

    values = []

    landmarks = (
        results.pose_landmarks.landmark
    )

    for index in POSE_INDICES:

        landmark = landmarks[index]

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility,
        ])

    return np.asarray(
        values,
        dtype=np.float32,
    )


# ============================================================
# EXTRACT RAW HAND
# ============================================================

def extract_hand(hand_landmarks):
    """
    Extract one MediaPipe hand.

    21 landmarks * xyz = 63 values.
    """

    if hand_landmarks is None:

        return np.zeros(
            63,
            dtype=np.float32,
        )

    values = []

    for landmark in (
        hand_landmarks.landmark
    ):

        values.extend([
            landmark.x,
            landmark.y,
            landmark.z,
        ])

    return np.asarray(
        values,
        dtype=np.float32,
    )


# ============================================================
# NORMALIZE HAND
# ============================================================

def normalize_hand(hand):
    """
    Normalize one hand.

    Translation:
        Wrist becomes origin.

    Scale:
        Wrist-to-middle-finger-MCP distance becomes
        approximately 1.

    MediaPipe hand landmark 0:
        wrist

    MediaPipe hand landmark 9:
        middle finger MCP
    """

    hand = np.asarray(
        hand,
        dtype=np.float32,
    ).reshape(21, 3)

    # Missing hand.
    if np.allclose(
        hand,
        0.0,
    ):

        return np.zeros(
            63,
            dtype=np.float32,
        )

    wrist = hand[0].copy()

    middle_mcp = hand[9].copy()

    scale = np.linalg.norm(
        middle_mcp - wrist
    )

    normalized = (
        hand - wrist
    )

    if scale > 1e-6:

        normalized = (
            normalized / scale
        )

    return normalized.flatten().astype(
        np.float32
    )


# ============================================================
# NORMALIZE POSE
# ============================================================

def normalize_pose(pose):
    """
    Normalize upper-body pose.

    Translation:
        Shoulder midpoint becomes origin.

    Scale:
        Shoulder-to-shoulder distance becomes
        approximately 1.

    Visibility values are preserved.
    """

    pose = np.asarray(
        pose,
        dtype=np.float32,
    ).reshape(6, 4)

    coordinates = (
        pose[:, :3].copy()
    )

    visibility = (
        pose[:, 3:4].copy()
    )

    # Missing pose.
    if np.allclose(
        coordinates,
        0.0,
    ):

        return np.zeros(
            24,
            dtype=np.float32,
        )

    left_shoulder = (
        coordinates[0]
    )

    right_shoulder = (
        coordinates[1]
    )

    center = (
        left_shoulder
        + right_shoulder
    ) / 2.0

    shoulder_distance = (
        np.linalg.norm(
            left_shoulder
            - right_shoulder
        )
    )

    coordinates = (
        coordinates - center
    )

    if shoulder_distance > 1e-6:

        coordinates = (
            coordinates
            / shoulder_distance
        )

    normalized = np.concatenate(
        [
            coordinates,
            visibility,
        ],
        axis=1,
    )

    return normalized.flatten().astype(
        np.float32
    )


# ============================================================
# FRAME FEATURE VECTOR
# ============================================================

def extract_frame_features(results):
    """
    Construct one 150-dimensional frame vector.

    Pose:
        6 landmarks * 4
        = 24

    Left hand:
        21 landmarks * 3
        = 63

    Right hand:
        21 landmarks * 3
        = 63

    Total:
        150
    """

    pose = extract_pose(
        results
    )

    left_hand = extract_hand(
        results.left_hand_landmarks
    )

    right_hand = extract_hand(
        results.right_hand_landmarks
    )

    pose = normalize_pose(
        pose
    )

    left_hand = normalize_hand(
        left_hand
    )

    right_hand = normalize_hand(
        right_hand
    )

    features = np.concatenate(
        [
            pose,
            left_hand,
            right_hand,
        ]
    ).astype(np.float32)

    if (
        features.shape
        != (FEATURES_PER_FRAME,)
    ):

        raise ValueError(
            "Expected "
            f"{FEATURES_PER_FRAME} "
            "features but got "
            f"{features.shape}."
        )

    return features


# ============================================================
# MOTION FEATURE INDICES
# ============================================================

def build_motion_coordinate_indices():
    """
    Determine which values participate in motion analysis.

    Pose visibility is intentionally excluded.

    Pose layout:
        6 * [x, y, z, visibility]

    Hands:
        xyz only.
    """

    indices = []

    for joint in range(6):

        base = joint * 4

        indices.extend([
            base,
            base + 1,
            base + 2,
        ])

    # All hand xyz coordinates.
    indices.extend(
        range(24, 150)
    )

    return np.asarray(
        indices,
        dtype=np.int32,
    )


MOTION_COORDINATE_INDICES = (
    build_motion_coordinate_indices()
)


# ============================================================
# CALCULATE LANDMARK MOTION
# ============================================================

def calculate_motion(sequence):
    """
    Calculate frame-to-frame motion from normalized
    pose/hand coordinates.

    Input:
        (N, 150)

    Output:
        (N,)
    """

    total_frames = len(
        sequence
    )

    if total_frames == 0:

        return np.zeros(
            0,
            dtype=np.float32,
        )

    if total_frames == 1:

        return np.zeros(
            1,
            dtype=np.float32,
        )

    coordinates = sequence[
        :,
        MOTION_COORDINATE_INDICES
    ]

    differences = np.abs(
        np.diff(
            coordinates,
            axis=0,
        )
    )

    frame_motion = np.mean(
        differences,
        axis=1,
    )

    # np.diff produces N-1 values.
    # Add zero for frame 0.
    frame_motion = np.concatenate(
        [
            np.zeros(
                1,
                dtype=np.float32,
            ),
            frame_motion.astype(
                np.float32
            ),
        ]
    )

    return frame_motion


# ============================================================
# SMOOTH MOTION
# ============================================================

def smooth_motion(motion):
    """
    Smooth the motion signal using a weighted
    five-frame kernel.
    """

    if len(motion) < 5:

        return motion.copy()

    kernel = np.asarray(
        [
            1,
            2,
            3,
            2,
            1,
        ],
        dtype=np.float32,
    )

    kernel /= np.sum(
        kernel
    )

    smoothed = np.convolve(
        motion,
        kernel,
        mode="same",
    )

    return smoothed.astype(
        np.float32
    )


# ============================================================
# DETECT ACTIVE SIGNING INTERVAL
# ============================================================

def detect_active_interval(sequence):
    """
    Detect the temporal region containing meaningful
    signing motion.

    Returns:
        start
        end
        metadata

    start/end are zero-based and inclusive.
    """

    total_frames = len(
        sequence
    )

    if total_frames == 0:

        return (
            0,
            0,
            {
                "fallback": True,
                "fallback_reason":
                    "empty_sequence",
                "peak_motion": 0.0,
                "threshold": 0.0,
                "raw_total_frames": 0,
                "active_start": 0,
                "active_end": 0,
                "active_length": 0,
            },
        )

    motion = calculate_motion(
        sequence
    )

    smoothed = smooth_motion(
        motion
    )

    peak_motion = float(
        np.max(smoothed)
    )

    threshold = max(
        peak_motion
        * MOTION_THRESHOLD_RATIO,
        MOTION_MIN_THRESHOLD,
    )

    active_indices = np.where(
        smoothed >= threshold
    )[0]

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if len(active_indices) < 2:

        return (
            0,
            total_frames - 1,
            {
                "fallback": True,
                "fallback_reason":
                    "insufficient_motion",
                "peak_motion":
                    peak_motion,
                "threshold":
                    threshold,
                "raw_total_frames":
                    total_frames,
                "active_start":
                    0,
                "active_end":
                    total_frames - 1,
                "active_length":
                    total_frames,
            },
        )

    start = int(
        active_indices[0]
    )

    end = int(
        active_indices[-1]
    )

    # --------------------------------------------------------
    # ADD TEMPORAL CONTEXT
    # --------------------------------------------------------

    detected_length = (
        end - start + 1
    )

    context_frames = max(
        2,
        int(
            round(
                detected_length
                * MOTION_CONTEXT_RATIO
            )
        ),
    )

    start = max(
        0,
        start - context_frames,
    )

    end = min(
        total_frames - 1,
        end + context_frames,
    )

    # --------------------------------------------------------
    # MINIMUM ACTIVE LENGTH
    # --------------------------------------------------------

    while (
        end - start + 1
        < MIN_ACTIVE_RAW_FRAMES
    ):

        expanded = False

        if start > 0:

            start -= 1
            expanded = True

        if (
            end - start + 1
            >= MIN_ACTIVE_RAW_FRAMES
        ):

            break

        if end < total_frames - 1:

            end += 1
            expanded = True

        if not expanded:

            break

    active_length = (
        end - start + 1
    )

    metadata = {
        "fallback":
            False,

        "fallback_reason":
            None,

        "peak_motion":
            peak_motion,

        "threshold":
            threshold,

        "raw_total_frames":
            total_frames,

        "active_start":
            start,

        "active_end":
            end,

        "active_length":
            active_length,

        "active_fraction":
            float(
                active_length
                / total_frames
            ),

        "context_frames":
            context_frames,
    }

    return (
        start,
        end,
        metadata,
    )


# ============================================================
# TEMPORAL RESAMPLING
# ============================================================

def resample_active_sequence(
    sequence,
    start,
    end,
):
    """
    Resample the detected active signing interval into
    exactly SEQUENCE_LENGTH frames.

    No interpolation of landmark values is performed.

    Instead, representative frames are selected
    uniformly from the active interval.
    """

    active_sequence = (
        sequence[
            start:end + 1
        ]
    )

    active_length = len(
        active_sequence
    )

    if active_length == 0:

        raise ValueError(
            "Active sequence contains zero frames."
        )

    if active_length == 1:

        return np.repeat(
            active_sequence,
            SEQUENCE_LENGTH,
            axis=0,
        ).astype(np.float32)

    sample_positions = np.linspace(
        0,
        active_length - 1,
        SEQUENCE_LENGTH,
    )

    sample_indices = np.round(
        sample_positions
    ).astype(np.int32)

    result = active_sequence[
        sample_indices
    ]

    return result.astype(
        np.float32
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    holistic,
):
    """
    V4 processing pipeline.

    Unlike V2, MediaPipe processes every decoded
    frame first.

    Motion detection is then performed on the full
    landmark sequence.

    The active signing interval is finally normalized
    to 30 frames.

    Returns:
        sequence:
            (30, 150)

        motion_metadata:
            dictionary describing temporal segmentation
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video: "
            f"{video_path}"
        )

    full_sequence = []

    try:

        while True:

            success, frame = (
                cap.read()
            )

            if not success:

                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb.flags.writeable = (
                False
            )

            results = holistic.process(
                rgb
            )

            features = (
                extract_frame_features(
                    results
                )
            )

            full_sequence.append(
                features
            )

    finally:

        cap.release()

    # --------------------------------------------------------
    # DECODE VALIDATION
    # --------------------------------------------------------

    if len(full_sequence) == 0:

        raise RuntimeError(
            "No frames decoded from: "
            f"{video_path}"
        )

    full_sequence = np.asarray(
        full_sequence,
        dtype=np.float32,
    )

    expected_raw_shape = (
        len(full_sequence),
        FEATURES_PER_FRAME,
    )

    if (
        full_sequence.shape
        != expected_raw_shape
    ):

        raise ValueError(
            "Invalid raw landmark sequence shape: "
            f"{full_sequence.shape}"
        )

    if np.isnan(
        full_sequence
    ).any():

        raise ValueError(
            "NaN detected in raw sequence: "
            f"{video_path.name}"
        )

    if np.isinf(
        full_sequence
    ).any():

        raise ValueError(
            "Infinity detected in raw sequence: "
            f"{video_path.name}"
        )

    # --------------------------------------------------------
    # ACTIVE SIGN DETECTION
    # --------------------------------------------------------

    (
        active_start,
        active_end,
        motion_metadata,
    ) = detect_active_interval(
        full_sequence
    )

    # --------------------------------------------------------
    # TEMPORAL NORMALIZATION
    # --------------------------------------------------------

    sequence = (
        resample_active_sequence(
            full_sequence,
            active_start,
            active_end,
        )
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    )

    if (
        sequence.shape
        != expected_shape
    ):

        raise ValueError(
            "Invalid V4 sequence shape: "
            f"{sequence.shape}. "
            f"Expected {expected_shape}."
        )

    if np.isnan(
        sequence
    ).any():

        raise ValueError(
            "NaN detected in V4 sequence: "
            f"{video_path.name}"
        )

    if np.isinf(
        sequence
    ).any():

        raise ValueError(
            "Infinity detected in V4 sequence: "
            f"{video_path.name}"
        )

    return (
        sequence,
        motion_metadata,
    )


# ============================================================
# FIND DATASET VIDEOS
# ============================================================

def discover_videos():
    """
    Discover all selected videos while preserving
    the existing train / val / test directories.
    """

    videos = []

    for split in SPLITS:

        for class_name in CLASSES:

            class_directory = (
                VIDEO_ROOT
                / split
                / class_name
            )

            if (
                not
                class_directory.exists()
            ):

                continue

            class_videos = sorted(
                class_directory.glob(
                    "*.mov"
                )
            )

            for video_path in (
                class_videos
            ):

                videos.append(
                    (
                        split,
                        class_name,
                        video_path,
                    )
                )

    return videos


# ============================================================
# CREATE METADATA RECORD
# ============================================================

def create_record(
    split,
    class_name,
    video_path,
    output_path,
    motion_metadata,
):
    """
    Create metadata for one V4 feature sequence.
    """

    return {
        "split":
            split,

        "class":
            class_name,

        "video":
            video_path.name,

        "feature_file":
            str(
                output_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "shape": [
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME,
        ],

        "motion": {
            key: (
                value.item()
                if isinstance(
                    value,
                    np.generic,
                )
                else value
            )
            for key, value
            in motion_metadata.items()
        },
    }


# ============================================================
# LOAD MOTION METADATA FOR RESUME
# ============================================================

def existing_motion_metadata(
    previous_records,
    output_path,
):
    """
    Recover motion metadata when a valid feature file
    is skipped during resume.

    If metadata is unavailable, return a marker rather
    than pretending the segmentation information exists.
    """

    relative = str(
        output_path.relative_to(
            PROJECT_ROOT
        )
    )

    record = previous_records.get(
        relative
    )

    if record is not None:

        motion = record.get(
            "motion"
        )

        if isinstance(
            motion,
            dict,
        ):

            return motion

    return {
        "fallback": None,
        "fallback_reason":
            "metadata_unavailable_during_resume",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 78
    )

    print(
        "FADHILI AI V4 - MOTION-AWARE "
        "MEDIAPIPE FEATURE EXTRACTION"
    )

    print(
        "=" * 78
    )

    print(
        f"Video root:     "
        f"{VIDEO_ROOT}"
    )

    print(
        f"Output root:    "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Sequence:       "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Features/frame: "
        f"{FEATURES_PER_FRAME}"
    )

    print(
        f"Motion ratio:   "
        f"{MOTION_THRESHOLD_RATIO}"
    )

    print(
        f"Min threshold:  "
        f"{MOTION_MIN_THRESHOLD}"
    )

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not VIDEO_ROOT.exists():

        print()

        print(
            "ERROR: Video directory "
            "does not exist:"
        )

        print(
            VIDEO_ROOT
        )

        sys.exit(1)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # METADATA PATH
    # --------------------------------------------------------

    metadata_path = (
        METADATA_ROOT
        / "fadhili_v4_features.json"
    )

    # --------------------------------------------------------
    # LOAD PREVIOUS V4 METADATA FOR RESUME
    # --------------------------------------------------------

    previous_records = {}

    if metadata_path.exists():

        try:

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                previous_metadata = (
                    json.load(file)
                )

            for record in (
                previous_metadata.get(
                    "records",
                    [],
                )
            ):

                feature_file = (
                    record.get(
                        "feature_file"
                    )
                )

                if feature_file:

                    previous_records[
                        feature_file
                    ] = record

        except Exception as error:

            print()

            print(
                "WARNING: Could not load "
                "previous V4 metadata:"
            )

            print(
                error
            )

    # --------------------------------------------------------
    # DISCOVER VIDEOS
    # --------------------------------------------------------

    videos = discover_videos()

    total = len(
        videos
    )

    print()

    print(
        f"Videos discovered: {total}"
    )

    if (
        total
        != EXPECTED_VIDEO_COUNT
    ):

        print()

        print(
            "WARNING:"
        )

        print(
            f"Expected "
            f"{EXPECTED_VIDEO_COUNT} "
            f"videos but found {total}."
        )

    if total == 0:

        print(
            "ERROR: No videos found."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # SPLIT COUNTS
    # --------------------------------------------------------

    split_counts = {
        split: 0
        for split in SPLITS
    }

    for (
        split,
        _,
        _,
    ) in videos:

        split_counts[
            split
        ] += 1

    print()

    print(
        "Dataset split:"
    )

    print(
        f"  Train: "
        f"{split_counts['train']}"
    )

    print(
        f"  Val:   "
        f"{split_counts['val']}"
    )

    print(
        f"  Test:  "
        f"{split_counts['test']}"
    )

    print()

    print(
        "Starting V4 extraction..."
    )

    print()

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    records = []

    processed = 0
    skipped = 0
    failed = 0

    fallbacks = 0

    active_lengths = []
    active_fractions = []

    failed_videos = []

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    with mp_holistic.Holistic(

        static_image_mode=False,

        model_complexity=1,

        smooth_landmarks=True,

        enable_segmentation=False,

        smooth_segmentation=False,

        refine_face_landmarks=False,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5,

    ) as holistic:

        # ----------------------------------------------------
        # PROCESS DATASET
        # ----------------------------------------------------

        for index, (
            split,
            class_name,
            video_path,
        ) in enumerate(
            videos,
            start=1,
        ):

            output_directory = (
                OUTPUT_ROOT
                / split
                / class_name
            )

            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                output_directory
                / (
                    video_path.stem
                    + ".npy"
                )
            )

            # =================================================
            # RESUME SUPPORT
            # =================================================

            if output_path.exists():

                try:

                    existing = np.load(
                        output_path,
                        mmap_mode="r",
                    )

                    valid_existing = (
                        existing.shape
                        == (
                            SEQUENCE_LENGTH,
                            FEATURES_PER_FRAME,
                        )
                        and
                        not np.isnan(
                            existing
                        ).any()
                        and
                        not np.isinf(
                            existing
                        ).any()
                    )

                except Exception:

                    valid_existing = False

                if valid_existing:

                    motion_metadata = (
                        existing_motion_metadata(
                            previous_records,
                            output_path,
                        )
                    )

                    print(
                        f"[{index}/{total}] "
                        "SKIP | "
                        f"{split.upper()} | "
                        f"{class_name} | "
                        f"{video_path.name}"
                    )

                    records.append(
                        create_record(
                            split,
                            class_name,
                            video_path,
                            output_path,
                            motion_metadata,
                        )
                    )

                    skipped += 1

                    continue

                print(
                    f"[{index}/{total}] "
                    "REBUILD INVALID | "
                    f"{split.upper()} | "
                    f"{class_name} | "
                    f"{video_path.name}"
                )

                try:

                    output_path.unlink()

                except OSError:

                    pass

            # =================================================
            # EXTRACT
            # =================================================

            print(
                f"[{index}/{total}] "
                f"{split.upper()} | "
                f"{class_name} | "
                f"{video_path.name}"
            )

            try:

                (
                    sequence,
                    motion_metadata,
                ) = process_video(
                    video_path,
                    holistic,
                )

                np.save(
                    output_path,
                    sequence,
                )

                records.append(
                    create_record(
                        split,
                        class_name,
                        video_path,
                        output_path,
                        motion_metadata,
                    )
                )

                processed += 1

                if (
                    motion_metadata.get(
                        "fallback"
                    )
                    is True
                ):

                    fallbacks += 1

                active_length = (
                    motion_metadata.get(
                        "active_length"
                    )
                )

                if (
                    active_length
                    is not None
                ):

                    active_lengths.append(
                        active_length
                    )

                active_fraction = (
                    motion_metadata.get(
                        "active_fraction"
                    )
                )

                if (
                    active_fraction
                    is not None
                ):

                    active_fractions.append(
                        active_fraction
                    )

            except Exception as error:

                failed += 1

                failed_videos.append(
                    {
                        "split":
                            split,

                        "class":
                            class_name,

                        "video":
                            video_path.name,

                        "error":
                            str(error),
                    }
                )

                print(
                    "    ERROR:",
                    error,
                )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "version":
            "fadhili-v4-motion-aware",

        "sequence_length":
            SEQUENCE_LENGTH,

        "features_per_frame":
            FEATURES_PER_FRAME,

        "expected_videos":
            EXPECTED_VIDEO_COUNT,

        "classes":
            CLASSES,

        "splits":
            SPLITS,

        "feature_definition": {

            "upper_body_pose": {
                "landmarks": 6,
                "indices":
                    POSE_INDICES,
                "values_per_landmark":
                    4,
                "features":
                    24,
            },

            "left_hand": {
                "landmarks":
                    21,
                "values_per_landmark":
                    3,
                "features":
                    63,
            },

            "right_hand": {
                "landmarks":
                    21,
                "values_per_landmark":
                    3,
                "features":
                    63,
            },
        },

        "normalization": {

            "hands":
                "wrist-centered and scaled "
                "by wrist-to-middle-MCP distance",

            "pose":
                "shoulder-midpoint centered and "
                "scaled by shoulder distance",

            "pose_visibility":
                "preserved but excluded from "
                "motion calculation",
        },

        "temporal_normalization": {

            "method":
                "landmark-motion active-window "
                "detection followed by uniform "
                "30-frame resampling",

            "motion_threshold_ratio":
                MOTION_THRESHOLD_RATIO,

            "minimum_motion_threshold":
                MOTION_MIN_THRESHOLD,

            "context_ratio":
                MOTION_CONTEXT_RATIO,

            "minimum_active_raw_frames":
                MIN_ACTIVE_RAW_FRAMES,

            "motion_coordinates":
                "pose xyz plus left/right hand xyz; "
                "pose visibility excluded",
        },

        "records":
            records,

        "failed_videos":
            failed_videos,
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ========================================================
    # FINAL FILE VALIDATION
    # ========================================================

    feature_files = list(
        OUTPUT_ROOT.rglob(
            "*.npy"
        )
    )

    valid_files = 0
    invalid_files = 0

    for feature_path in (
        feature_files
    ):

        try:

            array = np.load(
                feature_path,
                mmap_mode="r",
            )

            valid = (
                array.shape
                == (
                    SEQUENCE_LENGTH,
                    FEATURES_PER_FRAME,
                )
                and
                not np.isnan(
                    array
                ).any()
                and
                not np.isinf(
                    array
                ).any()
            )

            if valid:

                valid_files += 1

            else:

                invalid_files += 1

        except Exception:

            invalid_files += 1

    # ========================================================
    # REPORT
    # ========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "V4 FEATURE EXTRACTION FINISHED"
    )

    print(
        "=" * 78
    )

    print(
        f"Newly processed:      "
        f"{processed}"
    )

    print(
        f"Previously completed: "
        f"{skipped}"
    )

    print(
        f"Failed:               "
        f"{failed}"
    )

    print(
        f"Motion fallbacks:     "
        f"{fallbacks}"
    )

    print()

    print(
        f"Feature files found:  "
        f"{len(feature_files)}"
    )

    print(
        f"Valid feature files:  "
        f"{valid_files}"
    )

    print(
        f"Invalid feature files:"
        f" {invalid_files}"
    )

    print(
        f"Expected files:        "
        f"{EXPECTED_VIDEO_COUNT}"
    )

    print()

    print(
        "Expected sample shape:"
    )

    print(
        f"({SEQUENCE_LENGTH}, "
        f"{FEATURES_PER_FRAME})"
    )

    # --------------------------------------------------------
    # MOTION STATISTICS
    # --------------------------------------------------------

    if active_lengths:

        print()

        print(
            "Motion-aware segmentation:"
        )

        print(
            "  Mean active raw frames: "
            f"{np.mean(active_lengths):.2f}"
        )

        print(
            "  Min active raw frames:  "
            f"{np.min(active_lengths)}"
        )

        print(
            "  Max active raw frames:  "
            f"{np.max(active_lengths)}"
        )

    if active_fractions:

        print(
            "  Mean active fraction:   "
            f"{np.mean(active_fractions):.4f}"
        )

    print()

    print(
        "Metadata:"
    )

    print(
        metadata_path
    )

    # --------------------------------------------------------
    # FAILURES
    # --------------------------------------------------------

    if failed_videos:

        print()

        print(
            "FAILED VIDEOS:"
        )

        for item in (
            failed_videos
        ):

            print(
                " - "
                f"{item['split']}/"
                f"{item['class']}/"
                f"{item['video']}: "
                f"{item['error']}"
            )

    print()

    # --------------------------------------------------------
    # SUCCESS CHECK
    # --------------------------------------------------------

    if (
        valid_files
        == EXPECTED_VIDEO_COUNT
        and
        invalid_files == 0
        and
        failed == 0
    ):

        print(
            "SUCCESS: All 742 KSL videos "
            "have valid V4 motion-aware "
            "feature sequences."
        )

    else:

        print(
            "WARNING: V4 extraction is "
            "not yet complete."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()