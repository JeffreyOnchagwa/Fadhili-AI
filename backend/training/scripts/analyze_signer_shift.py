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
    / "fadhili_signer_shift_analysis.json"
)


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


def signer_id(video):

    match = re.search(
        r"Signer_(\d+)",
        video,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(video)

    return int(match.group(1))


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)


groups = {
    "development_01_12": [],
    "locked_13_15": [],
}

per_signer = defaultdict(list)
per_class_dev = defaultdict(list)
per_class_test = defaultdict(list)


for record in metadata["records"]:

    sid = signer_id(
        record["video"]
    )

    path = (
        PROJECT_ROOT
        / Path(record["feature_file"])
    )

    x = np.load(path).astype(
        np.float32
    )

    per_signer[sid].append(x)

    if sid <= 12:

        groups[
            "development_01_12"
        ].append(x)

        per_class_dev[
            record["class"]
        ].append(x)

    else:

        groups[
            "locked_13_15"
        ].append(x)

        per_class_test[
            record["class"]
        ].append(x)


def summarize(arrays):

    x = np.stack(arrays)

    return {
        "samples": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "mean_abs": float(
            np.mean(np.abs(x))
        ),
        "zero_fraction": float(
            np.mean(x == 0)
        ),
        "max_abs": float(
            np.max(np.abs(x))
        ),
    }


print("=" * 72)
print("GLOBAL DISTRIBUTION")
print("=" * 72)

global_results = {}

for name, arrays in groups.items():

    result = summarize(arrays)

    global_results[name] = result

    print()
    print(name)

    for key, value in result.items():
        print(
            f"  {key:<15}: {value}"
        )


print()
print("=" * 72)
print("PER-SIGNER LANDMARK STATISTICS")
print("=" * 72)

signer_results = {}

for sid in sorted(per_signer):

    result = summarize(
        per_signer[sid]
    )

    signer_results[str(sid)] = result

    print(
        f"Signer {sid:02d} | "
        f"N={result['samples']:>2} | "
        f"mean_abs={result['mean_abs']:.4f} | "
        f"std={result['std']:.4f} | "
        f"zeros={result['zero_fraction']:.2%} | "
        f"max={result['max_abs']:.2f}"
    )


print()
print("=" * 72)
print("CLASS DISTRIBUTION SHIFT")
print("=" * 72)

class_results = {}


for class_name in CLASS_NAMES:

    dev = np.stack(
        per_class_dev[class_name]
    )

    test = np.stack(
        per_class_test[class_name]
    )

    dev_flat = dev.reshape(
        len(dev),
        -1,
    )

    test_flat = test.reshape(
        len(test),
        -1,
    )

    dev_center = np.mean(
        dev_flat,
        axis=0,
    )

    test_center = np.mean(
        test_flat,
        axis=0,
    )

    centroid_distance = float(
        np.linalg.norm(
            dev_center
            - test_center
        )
    )

    dev_scale = float(
        np.mean(
            np.linalg.norm(
                dev_flat
                - dev_center,
                axis=1,
            )
        )
    )

    normalized_shift = (
        centroid_distance
        / (dev_scale + 1e-8)
    )

    result = {
        "development_samples":
            int(len(dev)),

        "test_samples":
            int(len(test)),

        "centroid_distance":
            centroid_distance,

        "development_spread":
            dev_scale,

        "normalized_shift":
            float(normalized_shift),

        "development_zero_fraction":
            float(
                np.mean(dev == 0)
            ),

        "test_zero_fraction":
            float(
                np.mean(test == 0)
            ),
    }

    class_results[
        class_name
    ] = result

    print()
    print(class_name)

    print(
        f"  centroid distance: "
        f"{centroid_distance:.4f}"
    )

    print(
        f"  normalized shift:  "
        f"{normalized_shift:.4f}"
    )

    print(
        f"  dev zeros:         "
        f"{result['development_zero_fraction']:.2%}"
    )

    print(
        f"  test zeros:        "
        f"{result['test_zero_fraction']:.2%}"
    )


output = {
    "global":
        global_results,

    "per_signer":
        signer_results,

    "per_class_shift":
        class_results,
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
print("=" * 72)
print("ANALYSIS COMPLETE")
print("=" * 72)

print(OUTPUT_PATH)