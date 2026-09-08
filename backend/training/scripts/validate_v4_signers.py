from collections import defaultdict

import mediapipe as mp
import numpy as np

from extract_landmarks_v4 import (
    VIDEO_ROOT,
    CLASSES,
    process_video,
)


def get_signer_number(filename):
    """
    Extract signer number from names such as:
    Signer_13_A356.mov
    """
    parts = filename.stem.split("_")

    if len(parts) < 2:
        return None

    try:
        return int(parts[1])
    except ValueError:
        return None


def find_one_video_per_signer():
    """
    Search the whole dataset and select one video
    for each signer 01-15.

    Spread classes across signers instead of accidentally
    testing Agreement for everyone.
    """

    signer_videos = defaultdict(list)

    for video in VIDEO_ROOT.rglob("*.mov"):

        signer = get_signer_number(video)

        if signer is not None:
            signer_videos[signer].append(video)

    selected = {}

    for signer in range(1, 16):

        videos = signer_videos.get(signer, [])

        if not videos:
            continue

        # Rotate desired class so we don't test only one class.
        desired_class = CLASSES[
            (signer - 1) % len(CLASSES)
        ]

        matching = [
            video
            for video in videos
            if video.parent.name == desired_class
        ]

        if matching:
            selected[signer] = sorted(matching)[0]
        else:
            selected[signer] = sorted(videos)[0]

    return selected


def main():

    selected = find_one_video_per_signer()

    print()
    print("=" * 90)
    print("FADHILI V4 — 15 SIGNER VALIDATION")
    print("=" * 90)

    print(f"Signers found: {len(selected)}/15")

    if len(selected) != 15:
        missing = [
            signer
            for signer in range(1, 16)
            if signer not in selected
        ]

        print("Missing signers:", missing)
        return

    results = []

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for signer in range(1, 16):

            video = selected[signer]

            print()
            print(
                f"Signer {signer:02d} | "
                f"{video.parent.name} | "
                f"{video.name}"
            )

            try:
                sequence, motion = process_video(
                    video,
                    holistic,
                )

                raw = motion["raw_total_frames"]
                start = motion["active_start"]
                end = motion["active_end"]
                active = motion["active_length"]
                fraction = motion["active_fraction"]
                fallback = motion["fallback"]

                valid_shape = (
                    sequence.shape == (30, 150)
                )

                finite = np.isfinite(
                    sequence
                ).all()

                print(
                    f"  raw={raw} "
                    f"start={start} "
                    f"end={end} "
                    f"active={active} "
                    f"fraction={fraction:.3f} "
                    f"fallback={fallback}"
                )

                print(
                    f"  shape={sequence.shape} "
                    f"finite={finite}"
                )

                results.append({
                    "signer": signer,
                    "class": video.parent.name,
                    "video": video.name,
                    "raw": raw,
                    "start": start,
                    "end": end,
                    "active": active,
                    "fraction": fraction,
                    "fallback": fallback,
                    "valid_shape": valid_shape,
                    "finite": finite,
                })

            except Exception as error:

                print(
                    f"  ERROR: {error}"
                )

                results.append({
                    "signer": signer,
                    "error": str(error),
                })

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    successful = [
        result
        for result in results
        if "error" not in result
    ]

    failed = [
        result
        for result in results
        if "error" in result
    ]

    if successful:

        fractions = np.array(
            [
                result["fraction"]
                for result in successful
            ],
            dtype=np.float32,
        )

        active_lengths = np.array(
            [
                result["active"]
                for result in successful
            ],
            dtype=np.float32,
        )

        print(
            f"Successful: {len(successful)}/15"
        )

        print(
            f"Failed:     {len(failed)}"
        )

        print(
            f"Fallbacks:  "
            f"{sum(r['fallback'] for r in successful)}"
        )

        print()
        print(
            "Active fraction:"
        )

        print(
            f"  mean = {fractions.mean():.3f}"
        )

        print(
            f"  min  = {fractions.min():.3f}"
        )

        print(
            f"  max  = {fractions.max():.3f}"
        )

        print()
        print(
            "Active frames:"
        )

        print(
            f"  mean = {active_lengths.mean():.1f}"
        )

        print(
            f"  min  = {active_lengths.min():.0f}"
        )

        print(
            f"  max  = {active_lengths.max():.0f}"
        )

        bad_shapes = [
            r
            for r in successful
            if not r["valid_shape"]
        ]

        bad_numbers = [
            r
            for r in successful
            if not r["finite"]
        ]

        print()
        print(
            f"Bad shapes: {len(bad_shapes)}"
        )

        print(
            f"NaN/Inf sequences: "
            f"{len(bad_numbers)}"
        )

    if failed:

        print()
        print("FAILED VIDEOS:")

        for result in failed:

            print(
                f"Signer {result['signer']:02d}: "
                f"{result['error']}"
            )

    print()
    print("=" * 90)

    if (
        len(successful) == 15
        and not failed
        and all(
            r["valid_shape"]
            and r["finite"]
            for r in successful
        )
    ):

        print(
            "PASS: V4 processed all "
            "15 signer samples successfully."
        )

    else:

        print(
            "FAIL: Review errors before "
            "full V4 extraction."
        )

    print("=" * 90)


if __name__ == "__main__":
    main()