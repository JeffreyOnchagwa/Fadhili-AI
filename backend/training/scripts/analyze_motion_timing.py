from pathlib import Path
import json
import re
from collections import defaultdict

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


def extract_signer(name):
    match = re.search(
        r"Signer_(\d+)",
        name,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(name)

    return int(match.group(1))


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)


per_signer = defaultdict(list)


# ============================================================
# MOTION SCORE
# ============================================================

def motion_profile(sequence):

    # Ignore pose visibility values:
    # pose layout is:
    # x,y,z,visibility repeated 6 times.
    #
    # Keep pose xyz + both hands xyz.

    pose_xyz_indices = []

    for joint in range(6):
        base = joint * 4

        pose_xyz_indices.extend([
            base,
            base + 1,
            base + 2,
        ])

    feature_indices = (
        pose_xyz_indices
        + list(range(24, 150))
    )

    coords = sequence[
        :,
        feature_indices
    ]

    differences = np.abs(
        np.diff(
            coords,
            axis=0,
        )
    )

    frame_motion = np.mean(
        differences,
        axis=1,
    )

    # np.diff gives 29 values.
    # Add zero for first frame.
    frame_motion = np.concatenate([
        np.zeros(1, dtype=np.float32),
        frame_motion,
    ])

    return frame_motion


# ============================================================
# LOAD FEATURES
# ============================================================

for record in metadata["records"]:

    sid = extract_signer(
        record["video"]
    )

    path = (
        PROJECT_ROOT
        / Path(record["feature_file"])
    )

    sequence = np.load(
        path
    ).astype(np.float32)

    profile = motion_profile(
        sequence
    )

    per_signer[sid].append(
        profile
    )


# ============================================================
# SUMMARIZE
# ============================================================

def summarize(profiles):

    profiles = np.stack(
        profiles
    )

    average_profile = np.mean(
        profiles,
        axis=0,
    )

    # Weighted center of motion
    positions = np.arange(
        30,
        dtype=np.float32,
    )

    total_motion = np.sum(
        profiles,
        axis=1,
    )

    centers = np.sum(
        profiles * positions,
        axis=1,
    ) / (
        total_motion + 1e-8
    )

    # Divide sequence into thirds
    early = np.mean(
        profiles[:, 0:10],
        axis=1,
    )

    middle = np.mean(
        profiles[:, 10:20],
        axis=1,
    )

    late = np.mean(
        profiles[:, 20:30],
        axis=1,
    )

    return {
        "mean_motion":
            float(np.mean(profiles)),

        "mean_motion_center_frame":
            float(np.mean(centers)),

        "early_motion":
            float(np.mean(early)),

        "middle_motion":
            float(np.mean(middle)),

        "late_motion":
            float(np.mean(late)),

        "average_profile":
            average_profile.tolist(),
    }


print("=" * 84)
print("FADHILI TEMPORAL MOTION ANALYSIS")
print("=" * 84)

signer_results = {}


for sid in sorted(per_signer):

    result = summarize(
        per_signer[sid]
    )

    signer_results[str(sid)] = result

    print(
        f"Signer {sid:02d} | "
        f"motion={result['mean_motion']:.4f} | "
        f"center={result['mean_motion_center_frame']:.2f} | "
        f"early={result['early_motion']:.4f} | "
        f"middle={result['middle_motion']:.4f} | "
        f"late={result['late_motion']:.4f}"
    )


# ============================================================
# GROUP COMPARISON
# ============================================================

dev_profiles = []

for sid in range(1, 13):
    dev_profiles.extend(
        per_signer[sid]
    )


diagnostic_profiles = []

for sid in range(13, 16):
    diagnostic_profiles.extend(
        per_signer[sid]
    )


dev = summarize(
    dev_profiles
)

diagnostic = summarize(
    diagnostic_profiles
)


print()
print("=" * 84)
print("GROUP COMPARISON")
print("=" * 84)

print()
print("Development 01-12")

print(
    f"  Mean motion: "
    f"{dev['mean_motion']:.4f}"
)

print(
    f"  Motion center: "
    f"{dev['mean_motion_center_frame']:.2f}"
)

print(
    f"  Early:  "
    f"{dev['early_motion']:.4f}"
)

print(
    f"  Middle: "
    f"{dev['middle_motion']:.4f}"
)

print(
    f"  Late:   "
    f"{dev['late_motion']:.4f}"
)


print()
print("Diagnostic 13-15")

print(
    f"  Mean motion: "
    f"{diagnostic['mean_motion']:.4f}"
)

print(
    f"  Motion center: "
    f"{diagnostic['mean_motion_center_frame']:.2f}"
)

print(
    f"  Early:  "
    f"{diagnostic['early_motion']:.4f}"
)

print(
    f"  Middle: "
    f"{diagnostic['middle_motion']:.4f}"
)

print(
    f"  Late:   "
    f"{diagnostic['late_motion']:.4f}"
)


# ============================================================
# PROFILE
# ============================================================

print()
print("=" * 84)
print("AVERAGE 30-FRAME MOTION PROFILE")
print("=" * 84)

print()
print("Frame    Development    Diagnostic")

for frame in range(30):

    print(
        f"{frame + 1:>2}       "
        f"{dev['average_profile'][frame]:>10.4f}    "
        f"{diagnostic['average_profile'][frame]:>10.4f}"
    )


print()
print("=" * 84)
print("ANALYSIS COMPLETE")
print("=" * 84)