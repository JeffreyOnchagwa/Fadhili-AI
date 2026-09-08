"""
Cache RAW MediaPipe landmarks for every KSL video, once.

Why this exists
---------------
The v2/v3/v4 pipeline ran MediaPipe and applied a fixed normalization in
the same pass, writing only the final 150-d feature vectors. That coupled
an expensive, irreversible step (MediaPipe inference, ~7 fps on this
2-core CPU) to a cheap, highly consequential one (normalization).

Diagnostics showed the normalization itself is a primary cause of the
studio -> in-the-wild accuracy collapse: normalize_hand() divides every
hand coordinate by the PROJECTED wrist-to-middle-MCP distance, which
collapses under foreshortening and inflates the whole hand block. Wild
signers 14/15 land at mean |x| = 0.75 against a studio range of
0.37-0.59, i.e. outside the training distribution entirely.

Caching raw landmarks decouples the two. After this runs once, any
normalization scheme can be evaluated in seconds instead of hours.

What is stored
--------------
Per video, one compressed .npz:

    pose    float16 (T, 33, 4)   x, y, z, visibility   image-normalized
    left    float16 (T, 21, 3)   x, y, z
    right   float16 (T, 21, 3)   x, y, z
    pose_ok bool    (T,)
    left_ok bool    (T,)
    right_ok bool   (T,)

ALL frames are kept (no temporal resampling), so sequence length,
sampling strategy and motion features all remain open questions that can
be revisited without re-running MediaPipe.

Coordinates are stored exactly as MediaPipe emits them. No normalization,
no aspect-ratio correction, no interpolation. Those are downstream
decisions.

Safety
------
- Never writes to data/features or data/features_v4 (v3/v4 inputs).
- Resumable: existing, valid .npz outputs are skipped.
- Each video is written atomically via a temp file + replace.
- A JSON checkpoint records per-video detection rates for auditing.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
VIDEO_ROOT = REPO / "backend" / "training" / "data" / "videos"
RAW_ROOT = REPO / "backend" / "training" / "data" / "raw_landmarks"
META_DIR = REPO / "backend" / "training" / "data" / "metadata"
CHECKPOINT = META_DIR / "raw_landmark_cache_index.json"

N_POSE = 33
N_HAND = 21


def build_task_list():
    """Every .mov under the video root, sorted for deterministic sharding."""
    tasks = []
    for path in sorted(VIDEO_ROOT.rglob("*.mov")):
        rel = path.relative_to(VIDEO_ROOT)
        out = (RAW_ROOT / rel).with_suffix(".npz")
        tasks.append((path, out))
    return tasks


def already_done(out_path):
    """True if a previous run wrote a readable, non-empty cache entry."""
    if not out_path.exists():
        return False
    try:
        with np.load(out_path) as data:
            return "pose" in data and data["pose"].shape[0] > 0
    except Exception:
        # Truncated or corrupt (e.g. killed mid-write) -> redo it.
        return False


def process_video(holistic, video_path):
    """
    Run MediaPipe Holistic over every frame of one video.

    Returns a dict of arrays, or None if the video could not be decoded.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    pose_seq, left_seq, right_seq = [], [], []
    pose_ok, left_ok, right_ok = [], [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)

        if res.pose_landmarks:
            pose_seq.append(
                [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in res.pose_landmarks.landmark
                ]
            )
            pose_ok.append(True)
        else:
            pose_seq.append(np.zeros((N_POSE, 4), dtype=np.float32))
            pose_ok.append(False)

        for landmarks, seq, flags in (
            (res.left_hand_landmarks, left_seq, left_ok),
            (res.right_hand_landmarks, right_seq, right_ok),
        ):
            if landmarks:
                seq.append(
                    [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
                )
                flags.append(True)
            else:
                seq.append(np.zeros((N_HAND, 3), dtype=np.float32))
                flags.append(False)

    fps = cap.get(5)  # CAP_PROP_FPS
    width = cap.get(3)
    height = cap.get(4)
    cap.release()

    if not pose_seq:
        return None

    return {
        "pose": np.asarray(pose_seq, dtype=np.float16),
        "left": np.asarray(left_seq, dtype=np.float16),
        "right": np.asarray(right_seq, dtype=np.float16),
        "pose_ok": np.asarray(pose_ok, dtype=bool),
        "left_ok": np.asarray(left_ok, dtype=bool),
        "right_ok": np.asarray(right_ok, dtype=bool),
        "meta": np.asarray(
            [fps, width, height, len(pose_seq)], dtype=np.float32
        ),
    }


def write_atomic(out_path, payload):
    """Write the npz via a temp file so an interrupt cannot corrupt it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(tmp, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--complexity",
        type=int,
        default=1,
        help="MediaPipe pose model_complexity.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N videos (0 = no limit). For smoke tests.",
    )
    args = parser.parse_args()

    # Keep each worker single-threaded so N workers actually scale on a
    # 2-core machine instead of oversubscribing it.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import mediapipe as mp

    tasks = build_task_list()
    shard = [
        t
        for i, t in enumerate(tasks)
        if i % args.num_workers == args.worker_id
    ]

    pending = [(v, o) for v, o in shard if not already_done(o)]
    if args.limit:
        pending = pending[: args.limit]

    tag = f"[w{args.worker_id}]"
    print(
        f"{tag} shard {len(shard)} videos, "
        f"{len(pending)} pending, complexity={args.complexity}",
        flush=True,
    )

    if not pending:
        print(f"{tag} nothing to do", flush=True)
        return

    index = {}
    started = time.time()
    done = 0

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for video_path, out_path in pending:
            try:
                payload = process_video(holistic, video_path)
            except Exception:
                print(
                    f"{tag} ERROR {video_path.name}\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )
                continue

            if payload is None:
                print(f"{tag} UNREADABLE {video_path.name}", flush=True)
                continue

            write_atomic(out_path, payload)

            rel = str(out_path.relative_to(RAW_ROOT)).replace("\\", "/")
            index[rel] = {
                "frames": int(payload["meta"][3]),
                "fps": float(payload["meta"][0]),
                "pose_rate": float(payload["pose_ok"].mean()),
                "left_rate": float(payload["left_ok"].mean()),
                "right_rate": float(payload["right_ok"].mean()),
            }

            done += 1
            if done % 10 == 0 or done == len(pending):
                elapsed = time.time() - started
                rate = done / elapsed
                remaining = (len(pending) - done) / rate if rate else 0
                print(
                    f"{tag} {done}/{len(pending)} "
                    f"({done / len(pending):.0%}) "
                    f"{rate * 60:.1f} vids/min "
                    f"ETA {remaining / 60:.0f} min",
                    flush=True,
                )

    # Merge this worker's index into the shared checkpoint.
    META_DIR.mkdir(parents=True, exist_ok=True)
    part = META_DIR / f"raw_landmark_cache_index.w{args.worker_id}.json"
    with part.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=1)

    print(
        f"{tag} COMPLETE {done} videos in "
        f"{(time.time() - started) / 60:.1f} min -> {part.name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
