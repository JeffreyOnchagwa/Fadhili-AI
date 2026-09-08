"""
Benchmark MediaPipe extraction strategies for Fadhili AI.

Purpose
-------
Signer-level diagnostics showed MediaPipe hand-detection failure rates
ranging from 6.7% (signer 08) to 62.4% (signer 11) using the v2/v3/v4
extraction settings (Holistic, model_complexity=1, full 1920x1080 frame).

Hands account for 126 of the 150 features, so a signer whose hands are
undetected in half the frames is effectively fed zeros. This benchmark
measures, per strategy, both the hand-detection rate and the wall-clock
cost so an informed re-extraction decision can be made on this hardware
(2-core i7-7Y75, no CUDA).

Strategies
----------
A  baseline      complexity=1, full frame                  (current v3)
B  complexity2   complexity=2, full frame
C  crop1         complexity=1, pose-guided upper-body crop
D  crop2         complexity=2, pose-guided upper-body crop

The crop strategies run a first Holistic pass to locate the signer, then
re-run on a tightened, upscaled upper-body region so the hands occupy a
far larger share of the input.

Nothing here writes to the feature cache; this script only reports.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

REPO = Path(__file__).resolve().parents[3]
VIDEO_ROOT = REPO / "backend" / "training" / "data" / "videos"
OUT_PATH = (
    REPO
    / "backend"
    / "training"
    / "data"
    / "metadata"
    / "extraction_benchmark.json"
)

# Pose landmarks used by the 150-feature representation.
POSE_IDS = [11, 12, 13, 14, 15, 16]


def iter_frames(path, max_frames=None):
    """Decode a video into a list of BGR frames."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def upper_body_box(pose_landmarks, width, height, margin=0.6):
    """
    Derive an upper-body crop box from pose landmarks.

    The box is anchored on the shoulders and expanded generously so that
    hands raised to head height or extended sideways stay inside it.
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks.landmark

    # Shoulders, elbows, wrists, plus nose for headroom.
    ids = [0, 11, 12, 13, 14, 15, 16]
    xs = [lm[i].x for i in ids]
    ys = [lm[i].y for i in ids]

    cx = (lm[11].x + lm[12].x) / 2.0
    shoulder_w = abs(lm[11].x - lm[12].x)

    if shoulder_w < 1e-4:
        return None

    # Half-width driven by shoulder span, widened by margin.
    half_w = shoulder_w * (1.0 + margin) * 1.6
    half_w = max(half_w, (max(xs) - min(xs)) / 2.0 + 0.08)

    x0 = cx - half_w
    x1 = cx + half_w

    y0 = min(ys) - 0.15
    y1 = max(ys) + 0.20

    # Clamp to frame, then convert to pixels.
    x0 = max(0.0, min(1.0, x0))
    x1 = max(0.0, min(1.0, x1))
    y0 = max(0.0, min(1.0, y0))
    y1 = max(0.0, min(1.0, y1))

    px0, px1 = int(x0 * width), int(x1 * width)
    py0, py1 = int(y0 * height), int(y1 * height)

    if px1 - px0 < 64 or py1 - py0 < 64:
        return None

    return px0, py0, px1, py1


def run_strategy(frames, complexity, crop, det=0.5, trk=0.5):
    """
    Run one extraction strategy over pre-decoded frames.

    Returns (left_hand_rate, right_hand_rate, pose_rate, seconds).
    """
    height, width = frames[0].shape[:2]

    box = None
    if crop:
        # Locate the signer once using a cheap pass over a few frames.
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=det,
            min_tracking_confidence=trk,
        ) as loc:
            boxes = []
            probe_ids = np.linspace(
                0, len(frames) - 1, min(8, len(frames))
            ).astype(int)
            for i in probe_ids:
                rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = loc.process(rgb)
                b = upper_body_box(res.pose_landmarks, width, height)
                if b:
                    boxes.append(b)

            if boxes:
                arr = np.array(boxes)
                # Union of probed boxes keeps all observed hand positions.
                box = (
                    int(arr[:, 0].min()),
                    int(arr[:, 1].min()),
                    int(arr[:, 2].max()),
                    int(arr[:, 3].max()),
                )

    left = right = pose = 0
    start = time.time()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=det,
        min_tracking_confidence=trk,
    ) as holistic:
        for frame in frames:
            img = frame
            if box:
                x0, y0, x1, y1 = box
                img = frame[y0:y1, x0:x1]
                # Upscale so hands occupy more pixels for the hand model.
                scale = 640.0 / max(img.shape[1], 1)
                if scale > 1.0:
                    img = cv2.resize(
                        img,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_LINEAR,
                    )

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            if res.left_hand_landmarks:
                left += 1
            if res.right_hand_landmarks:
                right += 1
            if res.pose_landmarks:
                pose += 1

    elapsed = time.time() - start
    n = len(frames)
    return left / n, right / n, pose / n, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos-per-signer",
        type=int,
        default=2,
        help="How many videos to benchmark per signer.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Cap decoded frames per video to bound runtime.",
    )
    parser.add_argument(
        "--signers",
        type=str,
        default="8,11,13,15",
        help="Comma-separated signer ids to benchmark.",
    )
    args = parser.parse_args()

    wanted = [int(s) for s in args.signers.split(",")]

    # Collect candidate videos per signer across all splits.
    by_signer = {s: [] for s in wanted}
    for path in VIDEO_ROOT.rglob("*.mov"):
        name = path.name
        if not name.startswith("Signer_"):
            continue
        try:
            sid = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if sid in by_signer:
            by_signer[sid].append(path)

    strategies = [
        ("A_baseline_c1_full", 1, False),
        ("B_c2_full", 2, False),
        ("C_c1_crop", 1, True),
        ("D_c2_crop", 2, True),
    ]

    results = {}

    for sid in wanted:
        vids = sorted(by_signer[sid])[: args.videos_per_signer]
        if not vids:
            print(f"signer {sid}: no videos found, skipping")
            continue

        print(f"\n=== Signer {sid:02d} ({len(vids)} videos) ===")
        results[str(sid)] = {}

        for video in vids:
            frames = iter_frames(video, args.max_frames)
            if not frames:
                continue

            print(f"  {video.name}  ({len(frames)} frames)")

            for label, complexity, crop in strategies:
                lh, rh, po, secs = run_strategy(frames, complexity, crop)

                entry = results[str(sid)].setdefault(
                    label,
                    {
                        "left": [],
                        "right": [],
                        "pose": [],
                        "fps": [],
                    },
                )
                entry["left"].append(lh)
                entry["right"].append(rh)
                entry["pose"].append(po)
                entry["fps"].append(len(frames) / secs)

                print(
                    f"    {label:<20} "
                    f"L={lh:5.1%} R={rh:5.1%} "
                    f"pose={po:5.1%} "
                    f"{len(frames) / secs:5.1f} fps"
                )

    # Aggregate.
    print("\n" + "=" * 72)
    print("AGGREGATE — mean hand-detection rate and throughput")
    print("=" * 72)
    print(f"{'strategy':<20}{'anyhand':>9}{'left':>9}{'right':>9}{'fps':>8}")

    summary = {}
    for label, _, _ in strategies:
        lefts, rights, fpss = [], [], []
        for sid in results:
            e = results[sid].get(label)
            if e:
                lefts += e["left"]
                rights += e["right"]
                fpss += e["fps"]
        if not lefts:
            continue
        ml, mr, mf = np.mean(lefts), np.mean(rights), np.mean(fpss)
        summary[label] = {
            "mean_left": float(ml),
            "mean_right": float(mr),
            "mean_hand": float((ml + mr) / 2),
            "mean_fps": float(mf),
        }
        print(
            f"{label:<20}{(ml + mr) / 2:>8.1%}"
            f"{ml:>9.1%}{mr:>9.1%}{mf:>8.1f}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {"per_signer": results, "summary": summary},
            f,
            indent=2,
        )

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
