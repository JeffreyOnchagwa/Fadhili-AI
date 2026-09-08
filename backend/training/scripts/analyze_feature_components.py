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


def get_signer(video_name):
    match = re.search(
        r"Signer_(\d+)",
        video_name,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(video_name)

    return int(match.group(1))


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)


signer_data = defaultdict(list)


for record in metadata["records"]:

    sid = get_signer(record["video"])

    path = (
        PROJECT_ROOT
        / Path(record["feature_file"])
    )

    x = np.load(path).astype(np.float32)

    if x.shape != (30, 150):
        raise ValueError(
            f"Unexpected shape {x.shape}: {path}"
        )

    signer_data[sid].append(x)


def component_stats(x):

    pose = x[:, :, 0:24]
    left = x[:, :, 24:87]
    right = x[:, :, 87:150]

    components = {
        "POSE": pose,
        "LEFT": left,
        "RIGHT": right,
    }

    output = {}

    for name, values in components.items():

        output[name] = {
            "mean_abs": float(
                np.mean(np.abs(values))
            ),
            "std": float(
                np.std(values)
            ),
            "zero_fraction": float(
                np.mean(values == 0)
            ),
        }

    return output


print("=" * 90)
print("PER-SIGNER FEATURE COMPONENT ANALYSIS")
print("=" * 90)

print(
    f"{'Signer':<8}"
    f"{'PoseAbs':>10}"
    f"{'Pose0%':>10}"
    f"{'LeftAbs':>10}"
    f"{'Left0%':>10}"
    f"{'RightAbs':>11}"
    f"{'Right0%':>10}"
)


for sid in sorted(signer_data):

    x = np.stack(
        signer_data[sid]
    )

    stats = component_stats(x)

    print(
        f"{sid:02d}{'':<6}"
        f"{stats['POSE']['mean_abs']:>10.3f}"
        f"{stats['POSE']['zero_fraction']:>9.1%}"
        f"{stats['LEFT']['mean_abs']:>10.3f}"
        f"{stats['LEFT']['zero_fraction']:>9.1%}"
        f"{stats['RIGHT']['mean_abs']:>11.3f}"
        f"{stats['RIGHT']['zero_fraction']:>9.1%}"
    )


print()
print("=" * 90)
print("DEVELOPMENT 01-12 VS LOCKED 13-15")
print("=" * 90)


dev = np.concatenate(
    [
        np.stack(signer_data[sid])
        for sid in range(1, 13)
    ],
    axis=0,
)

test = np.concatenate(
    [
        np.stack(signer_data[sid])
        for sid in range(13, 16)
    ],
    axis=0,
)


dev_stats = component_stats(dev)
test_stats = component_stats(test)


for component in [
    "POSE",
    "LEFT",
    "RIGHT",
]:

    print()
    print(component)

    print(
        f"  Development mean abs: "
        f"{dev_stats[component]['mean_abs']:.4f}"
    )

    print(
        f"  Test mean abs:        "
        f"{test_stats[component]['mean_abs']:.4f}"
    )

    print(
        f"  Development zeros:    "
        f"{dev_stats[component]['zero_fraction']:.2%}"
    )

    print(
        f"  Test zeros:           "
        f"{test_stats[component]['zero_fraction']:.2%}"
    )


print()
print("=" * 90)
print("COMPONENT ANALYSIS COMPLETE")
print("=" * 90)