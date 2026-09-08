from pathlib import Path
import json
import re

import numpy as np


# ============================================================
# PATHS / CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

METADATA_PATH = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "metadata"
    / "fadhili_v2_features.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "metadata"
    / "motion_segmentation_test.json"
)

SEQUENCE_LENGTH = 30

# Keep some context around the detected gesture.
CONTEXT_FRAMES = 2

# Fraction of peak smoothed motion used as threshold.
THRESHOLD_RATIO = 0.25

# Avoid extremely low thresholds.
MIN_THRESHOLD = 0.015


# ============================================================
# HELPERS
# ============================================================

def extract_signer(video_name):

    match = re.search(
        r"Signer_(\d+)",
        video_name,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(video_name)

    return int(match.group(1))


def coordinate_indices():

    # Pose:
    # 6 joints × [x, y, z, visibility]
    # Ignore visibility for motion calculation.

    indices = []

    for joint in range(6):

        base = joint * 4

        indices.extend([
            base,
            base + 1,
            base + 2,
        ])

    # Both hands are xyz only.
    indices.extend(
        range(24, 150)
    )

    return indices


COORDINATE_INDICES = coordinate_indices()


# ============================================================
# MOTION
# ============================================================

def calculate_motion(sequence):

    coords = sequence[
        :,
        COORDINATE_INDICES
    ]

    difference = np.abs(
        np.diff(
            coords,
            axis=0,
        )
    )

    motion = np.mean(
        difference,
        axis=1,
    )

    # Restore sequence length to 30.
    motion = np.concatenate([
        np.zeros(
            1,
            dtype=np.float32,
        ),
        motion.astype(np.float32),
    ])

    return motion


def smooth_motion(motion):

    # Small five-frame smoothing kernel.

    kernel = np.array(
        [1, 2, 3, 2, 1],
        dtype=np.float32,
    )

    kernel /= kernel.sum()

    return np.convolve(
        motion,
        kernel,
        mode="same",
    )


# ============================================================
# ACTIVE WINDOW
# ============================================================

def detect_active_window(sequence):

    motion = calculate_motion(
        sequence
    )

    smoothed = smooth_motion(
        motion
    )

    peak = float(
        np.max(smoothed)
    )

    threshold = max(
        peak * THRESHOLD_RATIO,
        MIN_THRESHOLD,
    )

    active = np.where(
        smoothed >= threshold
    )[0]

    # Fallback:
    # if almost no detectable motion exists,
    # preserve the complete sequence.

    if len(active) < 2:

        return {
            "start": 0,
            "end": SEQUENCE_LENGTH - 1,
            "length": SEQUENCE_LENGTH,
            "peak": peak,
            "threshold": threshold,
            "fallback": True,
            "motion": motion,
            "smoothed": smoothed,
        }

    start = int(
        active[0]
    )

    end = int(
        active[-1]
    )

    start = max(
        0,
        start - CONTEXT_FRAMES,
    )

    end = min(
        SEQUENCE_LENGTH - 1,
        end + CONTEXT_FRAMES,
    )

    return {
        "start": start,
        "end": end,
        "length": end - start + 1,
        "peak": peak,
        "threshold": threshold,
        "fallback": False,
        "motion": motion,
        "smoothed": smoothed,
    }


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    metadata = json.load(f)


# ============================================================
# SELECT ONE VIDEO PER SIGNER
# ============================================================

selected = {}

for record in metadata["records"]:

    sid = extract_signer(
        record["video"]
    )

    if sid not in selected:
        selected[sid] = record


if len(selected) != 15:

    raise RuntimeError(
        f"Expected 15 signers, found {len(selected)}"
    )


# ============================================================
# TEST
# ============================================================

print("=" * 88)
print("FADHILI MOTION SEGMENTATION TEST")
print("=" * 88)

print()
print(
    f"{'Signer':<8}"
    f"{'Class':<14}"
    f"{'Start':>7}"
    f"{'End':>7}"
    f"{'Length':>9}"
    f"{'Peak':>10}"
    f"{'Thresh':>10}"
    f"{'Fallback':>11}"
)

results = []


for sid in sorted(selected):

    record = selected[sid]

    feature_path = (
        PROJECT_ROOT
        / Path(record["feature_file"])
    )

    sequence = np.load(
        feature_path
    ).astype(np.float32)

    if sequence.shape != (
        30,
        150,
    ):
        raise ValueError(
            f"Unexpected shape: "
            f"{feature_path} "
            f"{sequence.shape}"
        )

    result = detect_active_window(
        sequence
    )

    print(
        f"{sid:02d}{'':<6}"
        f"{record['class']:<14}"
        f"{result['start'] + 1:>7}"
        f"{result['end'] + 1:>7}"
        f"{result['length']:>9}"
        f"{result['peak']:>10.4f}"
        f"{result['threshold']:>10.4f}"
        f"{str(result['fallback']):>11}"
    )

    results.append({
        "signer":
            sid,

        "class":
            record["class"],

        "video":
            record["video"],

        "feature_file":
            record["feature_file"],

        "start_frame_0_based":
            result["start"],

        "end_frame_0_based":
            result["end"],

        "start_frame_1_based":
            result["start"] + 1,

        "end_frame_1_based":
            result["end"] + 1,

        "window_length":
            result["length"],

        "peak_motion":
            result["peak"],

        "threshold":
            result["threshold"],

        "fallback":
            result["fallback"],

        "motion":
            result["motion"].tolist(),

        "smoothed_motion":
            result["smoothed"].tolist(),
    })


# ============================================================
# GROUP SUMMARY
# ============================================================

dev = [
    r
    for r in results
    if r["signer"] <= 12
]

diagnostic = [
    r
    for r in results
    if r["signer"] >= 13
]


def summarize(records):

    starts = np.array(
        [
            r["start_frame_1_based"]
            for r in records
        ],
        dtype=float,
    )

    ends = np.array(
        [
            r["end_frame_1_based"]
            for r in records
        ],
        dtype=float,
    )

    lengths = np.array(
        [
            r["window_length"]
            for r in records
        ],
        dtype=float,
    )

    return {
        "mean_start":
            float(np.mean(starts)),

        "mean_end":
            float(np.mean(ends)),

        "mean_length":
            float(np.mean(lengths)),

        "min_length":
            int(np.min(lengths)),

        "max_length":
            int(np.max(lengths)),

        "fallbacks":
            int(
                sum(
                    r["fallback"]
                    for r in records
                )
            ),
    }


dev_summary = summarize(dev)
diagnostic_summary = summarize(
    diagnostic
)


print()
print("=" * 88)
print("GROUP SUMMARY")
print("=" * 88)

print()
print("Development signers 01-12")

print(
    f"  Mean start:   "
    f"{dev_summary['mean_start']:.2f}"
)

print(
    f"  Mean end:     "
    f"{dev_summary['mean_end']:.2f}"
)

print(
    f"  Mean length:  "
    f"{dev_summary['mean_length']:.2f}"
)

print(
    f"  Length range: "
    f"{dev_summary['min_length']} - "
    f"{dev_summary['max_length']}"
)

print(
    f"  Fallbacks:    "
    f"{dev_summary['fallbacks']}"
)


print()
print("Diagnostic signers 13-15")

print(
    f"  Mean start:   "
    f"{diagnostic_summary['mean_start']:.2f}"
)

print(
    f"  Mean end:     "
    f"{diagnostic_summary['mean_end']:.2f}"
)

print(
    f"  Mean length:  "
    f"{diagnostic_summary['mean_length']:.2f}"
)

print(
    f"  Length range: "
    f"{diagnostic_summary['min_length']} - "
    f"{diagnostic_summary['max_length']}"
)

print(
    f"  Fallbacks:    "
    f"{diagnostic_summary['fallbacks']}"
)


# ============================================================
# WARNINGS
# ============================================================

very_short = [
    r
    for r in results
    if r["window_length"] < 8
]

almost_full = [
    r
    for r in results
    if r["window_length"] >= 28
]


print()
print("=" * 88)
print("SANITY CHECK")
print("=" * 88)

print(
    f"Very short windows (<8): "
    f"{len(very_short)}"
)

print(
    f"Almost-full windows (>=28): "
    f"{len(almost_full)}"
)


# ============================================================
# SAVE
# ============================================================

output = {
    "threshold_ratio":
        THRESHOLD_RATIO,

    "minimum_threshold":
        MIN_THRESHOLD,

    "context_frames":
        CONTEXT_FRAMES,

    "development_summary":
        dev_summary,

    "diagnostic_summary":
        diagnostic_summary,

    "samples":
        results,
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


print()
print("Saved:")
print(OUTPUT_PATH)

print()
print("=" * 88)
print("SEGMENTATION TEST COMPLETE")
print("=" * 88)