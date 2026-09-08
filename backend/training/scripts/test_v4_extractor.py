from pathlib import Path

import mediapipe as mp
import numpy as np

from extract_landmarks_v4 import (
    VIDEO_ROOT,
    process_video,
)


# ============================================================
# FIND ONE TEST VIDEO
# ============================================================

videos = sorted(
    VIDEO_ROOT.rglob("*.mov")
)

if not videos:
    raise RuntimeError(
        f"No videos found under {VIDEO_ROOT}"
    )

video_path = videos[0]

print("=" * 70)
print("FADHILI V4 SINGLE-VIDEO TEST")
print("=" * 70)

print()
print("Video:")
print(video_path)


# ============================================================
# RUN V4
# ============================================================

mp_holistic = mp.solutions.holistic

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

    sequence, motion = process_video(
        video_path,
        holistic,
    )


# ============================================================
# VALIDATE
# ============================================================

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"Sequence shape:   {sequence.shape}"
)

print(
    f"Data type:        {sequence.dtype}"
)

print(
    f"Contains NaN:     {np.isnan(sequence).any()}"
)

print(
    f"Contains Inf:     {np.isinf(sequence).any()}"
)

print(
    f"Raw frames:       {motion.get('raw_total_frames')}"
)

print(
    f"Active start:     {motion.get('active_start')}"
)

print(
    f"Active end:       {motion.get('active_end')}"
)

print(
    f"Active frames:    {motion.get('active_length')}"
)

print(
    f"Active fraction:  {motion.get('active_fraction')}"
)

print(
    f"Peak motion:      {motion.get('peak_motion')}"
)

print(
    f"Threshold:        {motion.get('threshold')}"
)

print(
    f"Fallback:         {motion.get('fallback')}"
)


# ============================================================
# HARD CHECKS
# ============================================================

assert sequence.shape == (
    30,
    150,
), sequence.shape

assert sequence.dtype == np.float32

assert not np.isnan(
    sequence
).any()

assert not np.isinf(
    sequence
).any()


print()
print("=" * 70)
print("PASS: V4 EXTRACTOR WORKS ON SINGLE VIDEO")
print("=" * 70)