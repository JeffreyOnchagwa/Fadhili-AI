from pathlib import Path
from collections import defaultdict
import json
import re

import numpy as np


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
    / "motion_segmentation_all_results.json"
)

SEQUENCE_LENGTH = 30
CONTEXT_FRAMES = 2
THRESHOLD_RATIO = 0.25
MIN_THRESHOLD = 0.015

CLASS_NAMES = [
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


def extract_signer(video_name):
    match = re.search(
        r"Signer_(\d+)",
        video_name,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(video_name)

    return int(match.group(1))


# Pose xyz only, excluding visibility.
COORDINATE_INDICES = []

for joint in range(6):
    base = joint * 4
    COORDINATE_INDICES.extend([
        base,
        base + 1,
        base + 2,
    ])

# Left + right hand xyz.
COORDINATE_INDICES.extend(
    range(24, 150)
)


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

    return np.concatenate([
        np.zeros(1, dtype=np.float32),
        motion.astype(np.float32),
    ])


def smooth_motion(motion):
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


def detect_window(sequence):
    motion = calculate_motion(sequence)
    smooth = smooth_motion(motion)

    peak = float(
        np.max(smooth)
    )

    threshold = max(
        peak * THRESHOLD_RATIO,
        MIN_THRESHOLD,
    )

    active = np.where(
        smooth >= threshold
    )[0]

    if len(active) < 2:
        return 0, 29, True

    start = max(
        0,
        int(active[0]) - CONTEXT_FRAMES,
    )

    end = min(
        29,
        int(active[-1]) + CONTEXT_FRAMES,
    )

    return start, end, False


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)


results = []
per_signer = defaultdict(list)
per_class = defaultdict(list)


for record in metadata["records"]:
    path = (
        PROJECT_ROOT
        / Path(record["feature_file"])
    )

    sequence = np.load(
        path
    ).astype(np.float32)

    if sequence.shape != (30, 150):
        raise ValueError(
            f"Bad shape: {path}: {sequence.shape}"
        )

    sid = extract_signer(
        record["video"]
    )

    start, end, fallback = (
        detect_window(sequence)
    )

    length = end - start + 1

    item = {
        "signer": sid,
        "class": record["class"],
        "video": record["video"],
        "start": start + 1,
        "end": end + 1,
        "length": length,
        "fallback": fallback,
    }

    results.append(item)
    per_signer[sid].append(item)
    per_class[record["class"]].append(item)


def summarize(items):
    starts = np.asarray(
        [x["start"] for x in items],
        dtype=float,
    )

    ends = np.asarray(
        [x["end"] for x in items],
        dtype=float,
    )

    lengths = np.asarray(
        [x["length"] for x in items],
        dtype=float,
    )

    return {
        "n": len(items),
        "mean_start": float(np.mean(starts)),
        "mean_end": float(np.mean(ends)),
        "mean_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "fallbacks": int(sum(
            x["fallback"]
            for x in items
        )),
        "short_under_8": int(sum(
            x["length"] < 8
            for x in items
        )),
        "almost_full_28_plus": int(sum(
            x["length"] >= 28
            for x in items
        )),
    }


print("=" * 86)
print("FADHILI — ALL-SEQUENCE MOTION SEGMENTATION")
print("=" * 86)

print()
print("PER-SIGNER")
print("-" * 86)

signer_summary = {}

for sid in sorted(per_signer):
    s = summarize(
        per_signer[sid]
    )

    signer_summary[str(sid)] = s

    print(
        f"Signer {sid:02d} | "
        f"N={s['n']:>2} | "
        f"start={s['mean_start']:>5.2f} | "
        f"end={s['mean_end']:>5.2f} | "
        f"length={s['mean_length']:>5.2f} | "
        f"range={s['min_length']:>2}-{s['max_length']:<2} | "
        f"fallback={s['fallbacks']}"
    )


print()
print("PER-CLASS")
print("-" * 86)

class_summary = {}

for class_name in CLASS_NAMES:
    s = summarize(
        per_class[class_name]
    )

    class_summary[class_name] = s

    print(
        f"{class_name:<12} | "
        f"N={s['n']:>2} | "
        f"start={s['mean_start']:>5.2f} | "
        f"end={s['mean_end']:>5.2f} | "
        f"length={s['mean_length']:>5.2f} | "
        f"range={s['min_length']:>2}-{s['max_length']:<2} | "
        f"fallback={s['fallbacks']}"
    )


dev_items = [
    x
    for x in results
    if x["signer"] <= 12
]

diagnostic_items = [
    x
    for x in results
    if x["signer"] >= 13
]

dev = summarize(dev_items)
diagnostic = summarize(
    diagnostic_items
)
overall = summarize(results)


print()
print("=" * 86)
print("GROUP COMPARISON")
print("=" * 86)

print()
print("Development 01-12")
print(
    f"  N:             {dev['n']}"
)
print(
    f"  Mean start:    {dev['mean_start']:.2f}"
)
print(
    f"  Mean end:      {dev['mean_end']:.2f}"
)
print(
    f"  Mean length:   {dev['mean_length']:.2f}"
)
print(
    f"  Fallbacks:     {dev['fallbacks']}"
)

print()
print("Diagnostic 13-15")
print(
    f"  N:             {diagnostic['n']}"
)
print(
    f"  Mean start:    {diagnostic['mean_start']:.2f}"
)
print(
    f"  Mean end:      {diagnostic['mean_end']:.2f}"
)
print(
    f"  Mean length:   {diagnostic['mean_length']:.2f}"
)
print(
    f"  Fallbacks:     {diagnostic['fallbacks']}"
)


print()
print("=" * 86)
print("GLOBAL SANITY CHECK")
print("=" * 86)

print(
    f"Total sequences:           {overall['n']}"
)

print(
    f"Fallbacks:                 {overall['fallbacks']}"
)

print(
    f"Very short windows (<8):   {overall['short_under_8']}"
)

print(
    f"Almost-full windows (>=28): "
    f"{overall['almost_full_28_plus']}"
)

print(
    f"Overall length range:      "
    f"{overall['min_length']} - "
    f"{overall['max_length']}"
)


output = {
    "config": {
        "threshold_ratio": THRESHOLD_RATIO,
        "minimum_threshold": MIN_THRESHOLD,
        "context_frames": CONTEXT_FRAMES,
    },
    "overall": overall,
    "development_01_12": dev,
    "diagnostic_13_15": diagnostic,
    "per_signer": signer_summary,
    "per_class": class_summary,
    "records": results,
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
print("=" * 86)

if (
    overall["fallbacks"] == 0
    and overall["short_under_8"] == 0
):
    print(
        "PASS: No pathological segmentation "
        "was detected."
    )
else:
    print(
        "WARNING: Review segmentation before "
        "regenerating features."
    )

print("=" * 86)