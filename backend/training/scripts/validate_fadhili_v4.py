from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "features_v4"
)

EXPECTED_SHAPE = (30, 150)
EXPECTED_TOTAL = 742

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


def get_signer(filename):

    parts = filename.stem.split("_")

    if len(parts) < 2:
        return None

    try:
        return int(parts[1])
    except ValueError:
        return None


def motion_score(sequence):

    # Exclude pose visibility:
    motion_indices = []

    for landmark in range(6):
        base = landmark * 4
        motion_indices.extend([
            base,
            base + 1,
            base + 2,
        ])

    motion_indices.extend(
        range(24, 150)
    )

    coords = sequence[:, motion_indices]

    diff = np.diff(
        coords,
        axis=0,
    )

    return float(
        np.sqrt(
            np.mean(diff ** 2)
        )
    )


def main():

    print()
    print("=" * 76)
    print("FADHILI AI V4 DATASET VALIDATION")
    print("=" * 76)

    files = sorted(
        FEATURE_ROOT.rglob("*.npy")
    )

    print(f"Feature files: {len(files)}")

    valid = 0
    invalid = 0

    signer_stats = defaultdict(
        lambda: {
            "count": 0,
            "motion": [],
            "zeros": [],
            "mean_abs": [],
        }
    )

    class_stats = defaultdict(
        lambda: {
            "count": 0,
            "motion": [],
        }
    )

    errors = []

    for index, path in enumerate(
        files,
        start=1,
    ):

        try:

            sequence = np.load(path)

            if sequence.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"bad shape {sequence.shape}"
                )

            if not np.isfinite(sequence).all():
                raise ValueError(
                    "NaN/Inf detected"
                )

            signer = get_signer(path)

            if signer is None:
                raise ValueError(
                    "could not determine signer"
                )

            class_name = path.parent.name

            motion = motion_score(
                sequence
            )

            zero_fraction = float(
                np.mean(sequence == 0)
            )

            mean_abs = float(
                np.mean(np.abs(sequence))
            )

            signer_stats[signer]["count"] += 1
            signer_stats[signer]["motion"].append(
                motion
            )
            signer_stats[signer]["zeros"].append(
                zero_fraction
            )
            signer_stats[signer]["mean_abs"].append(
                mean_abs
            )

            class_stats[class_name]["count"] += 1
            class_stats[class_name]["motion"].append(
                motion
            )

            valid += 1

        except Exception as error:

            invalid += 1

            errors.append(
                (
                    str(path),
                    str(error),
                )
            )

        if index % 100 == 0:
            print(
                f"Checked {index}/{len(files)}"
            )

    print()
    print("=" * 76)
    print("PER SIGNER")
    print("=" * 76)

    for signer in sorted(
        signer_stats
    ):

        stats = signer_stats[signer]

        mean_motion = np.mean(
            stats["motion"]
        )

        mean_zero = np.mean(
            stats["zeros"]
        )

        mean_abs = np.mean(
            stats["mean_abs"]
        )

        group = (
            "DEV"
            if signer <= 12
            else "DIAG"
        )

        print(
            f"{signer:02d} "
            f"{group:4s} "
            f"N={stats['count']:3d} "
            f"motion={mean_motion:.4f} "
            f"zero={mean_zero:.4f} "
            f"abs={mean_abs:.4f}"
        )

    print()
    print("=" * 76)
    print("DEV VS DIAGNOSTIC")
    print("=" * 76)

    dev_motion = []
    diag_motion = []

    dev_zero = []
    diag_zero = []

    dev_abs = []
    diag_abs = []

    for signer, stats in signer_stats.items():

        if signer <= 12:

            dev_motion.extend(
                stats["motion"]
            )

            dev_zero.extend(
                stats["zeros"]
            )

            dev_abs.extend(
                stats["mean_abs"]
            )

        else:

            diag_motion.extend(
                stats["motion"]
            )

            diag_zero.extend(
                stats["zeros"]
            )

            diag_abs.extend(
                stats["mean_abs"]
            )

    print(
        "DEV  "
        f"N={len(dev_motion)} "
        f"motion={np.mean(dev_motion):.4f} "
        f"zero={np.mean(dev_zero):.4f} "
        f"abs={np.mean(dev_abs):.4f}"
    )

    print(
        "DIAG "
        f"N={len(diag_motion)} "
        f"motion={np.mean(diag_motion):.4f} "
        f"zero={np.mean(diag_zero):.4f} "
        f"abs={np.mean(diag_abs):.4f}"
    )

    print()
    print("=" * 76)
    print("PER CLASS")
    print("=" * 76)

    for class_name in CLASSES:

        stats = class_stats.get(
            class_name
        )

        if not stats:
            print(
                f"{class_name:10s} MISSING"
            )
            continue

        print(
            f"{class_name:10s} "
            f"N={stats['count']:3d} "
            f"motion="
            f"{np.mean(stats['motion']):.4f}"
        )

    print()
    print("=" * 76)
    print("FINAL VALIDATION")
    print("=" * 76)

    print(
        f"Expected: {EXPECTED_TOTAL}"
    )

    print(
        f"Found:    {len(files)}"
    )

    print(
        f"Valid:    {valid}"
    )

    print(
        f"Invalid:  {invalid}"
    )

    print(
        f"Signers:  {len(signer_stats)}"
    )

    print(
        f"Classes:  {len(class_stats)}"
    )

    if errors:

        print()
        print("ERRORS:")

        for path, error in errors[:20]:

            print(
                f"{path}: {error}"
            )

    print()

    if (
        len(files) == EXPECTED_TOTAL
        and valid == EXPECTED_TOTAL
        and invalid == 0
        and len(signer_stats) == 15
        and len(class_stats) == 10
    ):

        print(
            "PASS: V4 DATASET IS READY FOR TRAINING."
        )

    else:

        print(
            "FAIL: V4 dataset requires review."
        )

    print("=" * 76)


if __name__ == "__main__":
    main()