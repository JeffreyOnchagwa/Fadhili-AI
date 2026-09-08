import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# CONFIGURATION
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

METADATA_PATH = (
    METADATA_ROOT
    / "fadhili_v4_features.json"
)

CHECKPOINT_PATH = (
    METADATA_ROOT
    / "fadhili_v4_extraction_checkpoint.json"
)

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 150
EXPECTED_VIDEO_COUNT = 742

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

POSE_INDICES = [
    11,
    12,
    13,
    14,
    15,
    16,
]

# Central 90% of cumulative motion energy
MOTION_LOW_QUANTILE = 0.05
MOTION_HIGH_QUANTILE = 0.95

CONTEXT_FRAMES = 5
MIN_ACTIVE_FRAMES = 8

mp_holistic = mp.solutions.holistic


# ============================================================
# POSE EXTRACTION
# ============================================================

def extract_pose(results):

    if results.pose_landmarks is None:
        return np.zeros(
            24,
            dtype=np.float32,
        )

    values = []

    for index in POSE_INDICES:

        landmark = (
            results.pose_landmarks.landmark[index]
        )

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
# HAND EXTRACTION
# ============================================================

def extract_hand(hand_landmarks):

    if hand_landmarks is None:
        return np.zeros(
            63,
            dtype=np.float32,
        )

    values = []

    for landmark in hand_landmarks.landmark:

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
# HAND NORMALIZATION
# ============================================================

def normalize_hand(hand):

    hand = np.asarray(
        hand,
        dtype=np.float32,
    ).reshape(21, 3)

    if np.allclose(hand, 0.0):

        return np.zeros(
            63,
            dtype=np.float32,
        )

    wrist = hand[0].copy()
    middle_mcp = hand[9].copy()

    scale = np.linalg.norm(
        middle_mcp - wrist
    )

    normalized = hand - wrist

    if scale > 1e-6:
        normalized = normalized / scale

    return (
        normalized
        .flatten()
        .astype(np.float32)
    )


# ============================================================
# POSE NORMALIZATION
# ============================================================

def normalize_pose(pose):

    pose = np.asarray(
        pose,
        dtype=np.float32,
    ).reshape(6, 4)

    coordinates = pose[:, :3].copy()
    visibility = pose[:, 3:4].copy()

    if np.allclose(coordinates, 0.0):

        return np.zeros(
            24,
            dtype=np.float32,
        )

    left_shoulder = coordinates[0]
    right_shoulder = coordinates[1]

    center = (
        left_shoulder
        + right_shoulder
    ) / 2.0

    shoulder_distance = np.linalg.norm(
        left_shoulder
        - right_shoulder
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

    return (
        normalized
        .flatten()
        .astype(np.float32)
    )


# ============================================================
# FRAME FEATURES
# ============================================================

def extract_frame_features(results):

    pose = normalize_pose(
        extract_pose(results)
    )

    left_hand = normalize_hand(
        extract_hand(
            results.left_hand_landmarks
        )
    )

    right_hand = normalize_hand(
        extract_hand(
            results.right_hand_landmarks
        )
    )

    features = np.concatenate(
        [
            pose,
            left_hand,
            right_hand,
        ]
    ).astype(np.float32)

    if features.shape != (
        FEATURES_PER_FRAME,
    ):

        raise ValueError(
            f"Expected {FEATURES_PER_FRAME} "
            f"features, got {features.shape}."
        )

    return features


# ============================================================
# MOTION INDICES
# ============================================================

def build_motion_indices():

    indices = []

    # Pose xyz only.
    # Visibility is intentionally excluded.
    for landmark_index in range(6):

        base = landmark_index * 4

        indices.extend([
            base,
            base + 1,
            base + 2,
        ])

    # Left + right hand xyz
    indices.extend(
        range(24, 150)
    )

    return np.asarray(
        indices,
        dtype=np.int32,
    )


MOTION_INDICES = build_motion_indices()


# ============================================================
# MOTION SIGNAL
# ============================================================

def calculate_motion(sequence):

    number_of_frames = (
        sequence.shape[0]
    )

    if number_of_frames <= 1:

        return np.zeros(
            number_of_frames,
            dtype=np.float32,
        )

    coordinates = sequence[
        :,
        MOTION_INDICES
    ]

    differences = np.diff(
        coordinates,
        axis=0,
    )

    motion = np.sqrt(
        np.mean(
            differences ** 2,
            axis=1,
        )
    )

    motion = np.concatenate(
        [
            np.zeros(
                1,
                dtype=np.float32,
            ),
            motion.astype(
                np.float32
            ),
        ]
    )

    return motion


# ============================================================
# SMOOTH MOTION
# ============================================================

def smooth_motion(motion):

    if motion.size < 5:
        return motion.copy()

    kernel = np.asarray(
        [1, 2, 3, 2, 1],
        dtype=np.float32,
    )

    kernel /= kernel.sum()

    return np.convolve(
        motion,
        kernel,
        mode="same",
    ).astype(np.float32)


# ============================================================
# ACTIVE INTERVAL
# ============================================================

def detect_active_interval(sequence):

    total_frames = (
        sequence.shape[0]
    )

    if total_frames <= 1:

        metadata = {
            "raw_total_frames":
                int(total_frames),

            "active_start":
                0,

            "active_end":
                max(
                    0,
                    total_frames - 1,
                ),

            "active_length":
                int(total_frames),

            "active_fraction":
                1.0,

            "fallback":
                True,

            "fallback_reason":
                "too_few_frames",

            "total_motion":
                0.0,
        }

        return (
            0,
            max(
                0,
                total_frames - 1,
            ),
            metadata,
        )

    motion = calculate_motion(
        sequence
    )

    motion = smooth_motion(
        motion
    )

    motion = np.maximum(
        motion,
        0.0,
    )

    total_motion = float(
        motion.sum()
    )

    if (
        not np.isfinite(total_motion)
        or total_motion <= 1e-8
    ):

        metadata = {
            "raw_total_frames":
                int(total_frames),

            "active_start":
                0,

            "active_end":
                int(total_frames - 1),

            "active_length":
                int(total_frames),

            "active_fraction":
                1.0,

            "fallback":
                True,

            "fallback_reason":
                "no_motion_energy",

            "total_motion":
                total_motion,
        }

        return (
            0,
            total_frames - 1,
            metadata,
        )

    cumulative = np.cumsum(
        motion
    )

    cumulative /= cumulative[-1]

    start = int(
        np.searchsorted(
            cumulative,
            MOTION_LOW_QUANTILE,
            side="left",
        )
    )

    end = int(
        np.searchsorted(
            cumulative,
            MOTION_HIGH_QUANTILE,
            side="left",
        )
    )

    start = max(
        0,
        start - CONTEXT_FRAMES,
    )

    end = min(
        total_frames - 1,
        end + CONTEXT_FRAMES,
    )

    while (
        end - start + 1
        < MIN_ACTIVE_FRAMES
    ):

        changed = False

        if start > 0:
            start -= 1
            changed = True

        if (
            end - start + 1
            >= MIN_ACTIVE_FRAMES
        ):
            break

        if end < (
            total_frames - 1
        ):
            end += 1
            changed = True

        if not changed:
            break

    active_length = (
        end - start + 1
    )

    metadata = {
        "raw_total_frames":
            int(total_frames),

        "active_start":
            int(start),

        "active_end":
            int(end),

        "active_length":
            int(active_length),

        "active_fraction":
            float(
                active_length
                / total_frames
            ),

        "fallback":
            False,

        "fallback_reason":
            None,

        "total_motion":
            total_motion,

        "motion_low_quantile":
            MOTION_LOW_QUANTILE,

        "motion_high_quantile":
            MOTION_HIGH_QUANTILE,

        "context_frames":
            CONTEXT_FRAMES,
    }

    return (
        start,
        end,
        metadata,
    )


# ============================================================
# RESAMPLE ACTIVE SEQUENCE
# ============================================================

def resample_sequence(
    sequence,
    start,
    end,
):

    active = sequence[
        start:end + 1
    ]

    number_of_frames = (
        active.shape[0]
    )

    if number_of_frames == 0:

        raise ValueError(
            "Detected active interval "
            "contains zero frames."
        )

    indices = np.linspace(
        0,
        number_of_frames - 1,
        SEQUENCE_LENGTH,
    )

    indices = np.round(
        indices
    ).astype(np.int32)

    sampled = active[
        indices
    ]

    return np.asarray(
        sampled,
        dtype=np.float32,
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    holistic,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    raw_sequence = []

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb.flags.writeable = False

            results = holistic.process(
                rgb
            )

            features = (
                extract_frame_features(
                    results
                )
            )

            raw_sequence.append(
                features
            )

    finally:

        cap.release()

    if not raw_sequence:

        raise RuntimeError(
            "No frames decoded from "
            f"{video_path.name}"
        )

    raw_sequence = np.asarray(
        raw_sequence,
        dtype=np.float32,
    )

    if (
        raw_sequence.ndim != 2
        or raw_sequence.shape[1]
        != FEATURES_PER_FRAME
    ):

        raise ValueError(
            "Invalid raw feature shape: "
            f"{raw_sequence.shape}"
        )

    if not np.isfinite(
        raw_sequence
    ).all():

        raise ValueError(
            "NaN or infinity detected in "
            f"{video_path.name}"
        )

    (
        start,
        end,
        motion_metadata,
    ) = detect_active_interval(
        raw_sequence
    )

    sequence = resample_sequence(
        raw_sequence,
        start,
        end,
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    )

    if sequence.shape != expected_shape:

        raise ValueError(
            "Invalid final sequence shape: "
            f"{sequence.shape}. "
            f"Expected {expected_shape}."
        )

    if not np.isfinite(
        sequence
    ).all():

        raise ValueError(
            "Invalid numeric values in "
            f"{video_path.name}"
        )

    return (
        sequence,
        motion_metadata,
    )


# ============================================================
# DISCOVER VIDEOS
# ============================================================

def discover_videos():

    videos = []

    for split in SPLITS:

        for class_name in CLASSES:

            directory = (
                VIDEO_ROOT
                / split
                / class_name
            )

            if not directory.exists():
                continue

            for video_path in sorted(
                directory.glob("*.mov")
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
# CHECK EXISTING FEATURE
# ============================================================

def valid_existing_feature(
    output_path,
):

    if not output_path.exists():
        return False

    try:

        array = np.load(
            output_path,
            mmap_mode="r",
        )

        if array.shape != (
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME,
        ):
            return False

        if not np.isfinite(
            array
        ).all():
            return False

        return True

    except Exception:
        return False


# ============================================================
# RECORD KEY
# ============================================================

def record_key(
    split,
    class_name,
    video_name,
):

    return (
        f"{split}|"
        f"{class_name}|"
        f"{video_name}"
    )


# ============================================================
# LOAD PREVIOUS METADATA / CHECKPOINT
# ============================================================

def load_previous_records():

    records = {}

    possible_paths = [
        CHECKPOINT_PATH,
        METADATA_PATH,
    ]

    for path in possible_paths:

        if not path.exists():
            continue

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            for record in data.get(
                "records",
                []
            ):

                key = record_key(
                    record["split"],
                    record["class"],
                    record["video"],
                )

                records[key] = record

        except Exception as error:

            print(
                f"WARNING: Could not read "
                f"{path.name}: {error}"
            )

    return records


# ============================================================
# CREATE RECORD
# ============================================================

def create_record(
    split,
    class_name,
    video_path,
    output_path,
    motion_metadata=None,
):

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

        "motion":
            motion_metadata,
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    records,
    failed_videos,
):

    data = {
        "version":
            "fadhili-v4-motion-aware",

        "status":
            "in_progress",

        "sequence_length":
            SEQUENCE_LENGTH,

        "features_per_frame":
            FEATURES_PER_FRAME,

        "records":
            list(
                records.values()
            ),

        "failed_videos":
            failed_videos,
    }

    temporary_path = (
        CHECKPOINT_PATH.with_suffix(
            ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )

    temporary_path.replace(
        CHECKPOINT_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "FADHILI AI V4 - RESUMABLE "
        "MOTION-AWARE EXTRACTION"
    )
    print("=" * 72)

    print(
        f"Video root:  {VIDEO_ROOT}"
    )

    print(
        f"Output root: {OUTPUT_ROOT}"
    )

    if not VIDEO_ROOT.exists():

        print(
            "ERROR: Video root does not exist."
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

    videos = discover_videos()

    total = len(videos)

    print()
    print(
        f"Videos discovered: {total}"
    )

    if total == 0:

        print(
            "ERROR: No videos discovered."
        )

        sys.exit(1)

    if total != EXPECTED_VIDEO_COUNT:

        print(
            f"WARNING: Expected "
            f"{EXPECTED_VIDEO_COUNT}, "
            f"found {total}."
        )

    previous_records = (
        load_previous_records()
    )

    records = dict(
        previous_records
    )

    failed_videos = []

    completed = 0
    skipped = 0
    newly_processed = 0
    failed = 0

    # Count existing valid files before starting.
    existing_valid = 0

    for (
        split,
        class_name,
        video_path,
    ) in videos:

        output_path = (
            OUTPUT_ROOT
            / split
            / class_name
            / (
                video_path.stem
                + ".npy"
            )
        )

        if valid_existing_feature(
            output_path
        ):
            existing_valid += 1

    print(
        f"Already completed: {existing_valid}"
    )

    print(
        f"Remaining:         "
        f"{total - existing_valid}"
    )

    print()
    print(
        "Existing valid files will be "
        "SKIPPED automatically."
    )
    print()

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

            key = record_key(
                split,
                class_name,
                video_path.name,
            )

            print(
                f"[{index}/{total}] "
                f"{split.upper()} | "
                f"{class_name} | "
                f"{video_path.name}"
            )

            # ================================================
            # RESUME: SKIP VALID EXISTING OUTPUT
            # ================================================

            if valid_existing_feature(
                output_path
            ):

                print(
                    "    SKIP: already completed"
                )

                skipped += 1
                completed += 1

                # Preserve old motion metadata if available.
                if key not in records:

                    records[key] = (
                        create_record(
                            split,
                            class_name,
                            video_path,
                            output_path,
                            motion_metadata=None,
                        )
                    )

                continue

            # Remove corrupt/partial output.
            if output_path.exists():

                print(
                    "    Existing output invalid; "
                    "reprocessing."
                )

                try:
                    output_path.unlink()
                except Exception:
                    pass

            # ================================================
            # PROCESS MISSING VIDEO
            # ================================================

            try:

                (
                    sequence,
                    motion_metadata,
                ) = process_video(
                    video_path,
                    holistic,
                )

                # Atomic-ish save:
                # save temporary .npy then rename.
                temporary_output = (
                    output_path.parent
                    / (
                        output_path.stem
                        + ".partial.npy"
                    )
                )

                np.save(
                    temporary_output,
                    sequence,
                )

                # Validate before replacing final output.
                test_array = np.load(
                    temporary_output,
                    mmap_mode="r",
                )

                if (
                    test_array.shape
                    != (
                        SEQUENCE_LENGTH,
                        FEATURES_PER_FRAME,
                    )
                ):

                    raise ValueError(
                        "Temporary feature file "
                        "has invalid shape."
                    )

                if not np.isfinite(
                    test_array
                ).all():

                    raise ValueError(
                        "Temporary feature file "
                        "contains NaN/Inf."
                    )

                del test_array

                temporary_output.replace(
                    output_path
                )

                records[key] = (
                    create_record(
                        split,
                        class_name,
                        video_path,
                        output_path,
                        motion_metadata,
                    )
                )

                newly_processed += 1
                completed += 1

                print(
                    "    DONE"
                )

                # Save progress after every successful video.
                save_checkpoint(
                    records,
                    failed_videos,
                )

            except Exception as error:

                failed += 1

                print(
                    f"    ERROR: {error}"
                )

                failed_videos.append({
                    "split":
                        split,

                    "class":
                        class_name,

                    "video":
                        video_path.name,

                    "error":
                        str(error),
                })

                # Clean incomplete temporary output.
                partial = (
                    output_path.parent
                    / (
                        output_path.stem
                        + ".partial.npy"
                    )
                )

                if partial.exists():

                    try:
                        partial.unlink()
                    except Exception:
                        pass

                save_checkpoint(
                    records,
                    failed_videos,
                )

    # ========================================================
    # FINAL FILE VALIDATION
    # ========================================================

    print()
    print(
        "Running final validation..."
    )

    valid = 0
    invalid = 0
    missing = 0

    for (
        split,
        class_name,
        video_path,
    ) in videos:

        output_path = (
            OUTPUT_ROOT
            / split
            / class_name
            / (
                video_path.stem
                + ".npy"
            )
        )

        if not output_path.exists():

            missing += 1
            continue

        if valid_existing_feature(
            output_path
        ):
            valid += 1
        else:
            invalid += 1

    # ========================================================
    # FINAL METADATA
    # ========================================================

    metadata = {
        "version":
            "fadhili-v4-motion-aware",

        "status":
            "complete"
            if (
                valid == total
                and invalid == 0
                and missing == 0
            )
            else "incomplete",

        "sequence_length":
            SEQUENCE_LENGTH,

        "features_per_frame":
            FEATURES_PER_FRAME,

        "classes":
            CLASSES,

        "splits":
            SPLITS,

        "temporal_method":
            (
                "full-video landmark extraction; "
                "central 90 percent cumulative "
                "motion-energy interval; "
                "5-frame context; "
                "30-frame resampling"
            ),

        "motion_energy_quantiles": {
            "low":
                MOTION_LOW_QUANTILE,

            "high":
                MOTION_HIGH_QUANTILE,
        },

        "context_frames":
            CONTEXT_FRAMES,

        "feature_definition": {
            "pose":
                (
                    "6 upper-body landmarks x "
                    "xyz+visibility = 24"
                ),

            "left_hand":
                "21 landmarks x xyz = 63",

            "right_hand":
                "21 landmarks x xyz = 63",
        },

        "normalization": {
            "pose":
                (
                    "shoulder-midpoint centered "
                    "and shoulder-distance scaled"
                ),

            "hands":
                (
                    "wrist centered and "
                    "wrist-to-middle-MCP scaled"
                ),
        },

        "resume_statistics": {
            "skipped_existing":
                skipped,

            "newly_processed":
                newly_processed,

            "processing_failures":
                failed,
        },

        "validation": {
            "expected":
                total,

            "valid":
                valid,

            "invalid":
                invalid,

            "missing":
                missing,
        },

        "records":
            list(
                records.values()
            ),

        "failed_videos":
            failed_videos,
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print(
        "V4 EXTRACTION FINISHED"
    )
    print("=" * 72)

    print(
        f"Videos discovered:  {total}"
    )

    print(
        f"Skipped existing:   {skipped}"
    )

    print(
        f"Newly processed:    {newly_processed}"
    )

    print(
        f"Processing failed:  {failed}"
    )

    print()
    print(
        f"Valid feature files: {valid}"
    )

    print(
        f"Invalid files:       {invalid}"
    )

    print(
        f"Missing files:       {missing}"
    )

    print()
    print(
        f"Metadata: {METADATA_PATH}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print()

    if (
        valid == total
        and invalid == 0
        and missing == 0
    ):

        print(
            "PASS: ALL V4 FEATURES "
            "ARE PRESENT AND VALID."
        )

    else:

        print(
            "INCOMPLETE: rerun this same "
            "command to retry missing/"
            "invalid videos."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()