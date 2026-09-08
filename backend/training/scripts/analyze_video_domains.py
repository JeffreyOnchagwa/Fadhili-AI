from pathlib import Path
from collections import defaultdict
import json
import re

import cv2
import numpy as np


# ============================================================
# PATHS
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

VIDEO_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "videos"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "metadata"
    / "fadhili_video_domain_analysis.json"
)


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
        raise ValueError(
            f"Could not extract signer: {video_name}"
        )

    return int(match.group(1))


def find_video(record):

    split = record["split"]
    class_name = record["class"]
    video_name = record["video"]

    expected = (
        VIDEO_ROOT
        / split
        / class_name
        / video_name
    )

    if expected.exists():
        return expected

    # Handles filenames where Kaggle preserved URL encoding.
    folder = (
        VIDEO_ROOT
        / split
        / class_name
    )

    if not folder.exists():
        return None

    signer = extract_signer(video_name)

    candidates = []

    for path in folder.iterdir():

        if not path.is_file():
            continue

        try:
            candidate_signer = extract_signer(
                path.name
            )
        except ValueError:
            continue

        if candidate_signer == signer:
            candidates.append(path)

    # We don't guess when multiple files could match.
    return None


def inspect_video(path):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return None

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    duration = (
        frames / fps
        if fps > 0
        else None
    )

    aspect_ratio = (
        width / height
        if height > 0
        else None
    )

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "size_mb": (
            path.stat().st_size
            / (1024 * 1024)
        ),
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


per_signer = defaultdict(list)

missing = []
failed = []


# ============================================================
# INSPECT ALL VIDEOS
# ============================================================

print("=" * 78)
print("FADHILI VIDEO DOMAIN ANALYSIS")
print("=" * 78)

for index, record in enumerate(
    metadata["records"],
    start=1,
):

    path = find_video(record)

    if path is None:

        missing.append({
            "class": record["class"],
            "video": record["video"],
        })

        continue

    info = inspect_video(path)

    if info is None:

        failed.append(str(path))
        continue

    sid = extract_signer(
        record["video"]
    )

    info["class"] = record["class"]
    info["video"] = record["video"]
    info["path"] = str(path)

    per_signer[sid].append(info)

    if index % 50 == 0:
        print(
            f"Inspected {index}/"
            f"{len(metadata['records'])}"
        )


# ============================================================
# SUMMARIZE SIGNER
# ============================================================

def summarize(videos):

    fps = np.asarray(
        [x["fps"] for x in videos],
        dtype=float,
    )

    frames = np.asarray(
        [x["frames"] for x in videos],
        dtype=float,
    )

    durations = np.asarray(
        [
            x["duration"]
            for x in videos
            if x["duration"] is not None
        ],
        dtype=float,
    )

    sizes = np.asarray(
        [x["size_mb"] for x in videos],
        dtype=float,
    )

    resolutions = defaultdict(int)

    for x in videos:

        key = (
            f"{x['width']}x{x['height']}"
        )

        resolutions[key] += 1

    return {
        "videos": len(videos),

        "mean_fps":
            float(np.mean(fps)),

        "min_fps":
            float(np.min(fps)),

        "max_fps":
            float(np.max(fps)),

        "mean_frames":
            float(np.mean(frames)),

        "mean_duration":
            float(np.mean(durations)),

        "min_duration":
            float(np.min(durations)),

        "max_duration":
            float(np.max(durations)),

        "mean_size_mb":
            float(np.mean(sizes)),

        "resolutions":
            dict(resolutions),
    }


# ============================================================
# PRINT PER-SIGNER RESULTS
# ============================================================

print()
print("=" * 78)
print("PER-SIGNER VIDEO STATISTICS")
print("=" * 78)

signer_results = {}


for sid in sorted(per_signer):

    result = summarize(
        per_signer[sid]
    )

    signer_results[str(sid)] = result

    resolution_text = ", ".join(
        f"{resolution}:{count}"
        for resolution, count
        in result["resolutions"].items()
    )

    print()
    print(f"Signer {sid:02d}")

    print(
        f"  Videos:        "
        f"{result['videos']}"
    )

    print(
        f"  FPS mean:      "
        f"{result['mean_fps']:.2f}"
    )

    print(
        f"  FPS range:     "
        f"{result['min_fps']:.2f} - "
        f"{result['max_fps']:.2f}"
    )

    print(
        f"  Mean frames:   "
        f"{result['mean_frames']:.1f}"
    )

    print(
        f"  Mean duration: "
        f"{result['mean_duration']:.2f}s"
    )

    print(
        f"  Duration range:"
        f" {result['min_duration']:.2f}s - "
        f"{result['max_duration']:.2f}s"
    )

    print(
        f"  Mean size:     "
        f"{result['mean_size_mb']:.2f} MB"
    )

    print(
        f"  Resolutions:   "
        f"{resolution_text}"
    )


# ============================================================
# GROUP COMPARISON
# ============================================================

development_videos = []

for sid in range(1, 13):
    development_videos.extend(
        per_signer[sid]
    )


diagnostic_videos = []

for sid in range(13, 16):
    diagnostic_videos.extend(
        per_signer[sid]
    )


dev = summarize(
    development_videos
)

diagnostic = summarize(
    diagnostic_videos
)


print()
print("=" * 78)
print("01-12 VS 13-15")
print("=" * 78)

print()
print("Development signers 01-12")

print(
    f"  Videos:        "
    f"{dev['videos']}"
)

print(
    f"  Mean FPS:      "
    f"{dev['mean_fps']:.2f}"
)

print(
    f"  Mean frames:   "
    f"{dev['mean_frames']:.1f}"
)

print(
    f"  Mean duration: "
    f"{dev['mean_duration']:.2f}s"
)

print(
    f"  Mean size:     "
    f"{dev['mean_size_mb']:.2f} MB"
)

print(
    f"  Resolutions:   "
    f"{dev['resolutions']}"
)


print()
print("Diagnostic signers 13-15")

print(
    f"  Videos:        "
    f"{diagnostic['videos']}"
)

print(
    f"  Mean FPS:      "
    f"{diagnostic['mean_fps']:.2f}"
)

print(
    f"  Mean frames:   "
    f"{diagnostic['mean_frames']:.1f}"
)

print(
    f"  Mean duration: "
    f"{diagnostic['mean_duration']:.2f}s"
)

print(
    f"  Mean size:     "
    f"{diagnostic['mean_size_mb']:.2f} MB"
)

print(
    f"  Resolutions:   "
    f"{diagnostic['resolutions']}"
)


# ============================================================
# SAVE
# ============================================================

output = {
    "per_signer": signer_results,
    "development_01_12": dev,
    "diagnostic_13_15": diagnostic,
    "missing_videos": missing,
    "failed_videos": failed,
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
print("=" * 78)
print("FILE CHECK")
print("=" * 78)

print(
    f"Missing videos: {len(missing)}"
)

print(
    f"Failed videos:  {len(failed)}"
)

print()
print("Results saved:")
print(OUTPUT_PATH)

print()
print("=" * 78)
print("ANALYSIS COMPLETE")
print("=" * 78)