"""
Download the remaining NAMED KSL classes from the eKitabu Kaggle dataset.

Vocabulary policy
-----------------
The dataset contains 30 class directories, but only 15 carry a real
glossed label:

    Agreement, Apple, Colour, Friend, Gift, Market, Monday, Picture,
    Proud, Sweater, Teach, Tomatoes, Tortoise, Twin, Ugali

The other 15 are opaque identifiers (No_9, No_17, No_22, No_35, No_48,
No_54, No_66, No_73, No_89, No_91, No_100, No_125, No_268, No_388,
No_444). Their meanings are not documented anywhere in the dataset, so
they must NEVER be surfaced as product vocabulary or guessed at. They
remain eligible only as unlabelled/auxiliary data for representation
learning, where the gloss is irrelevant.

This script therefore downloads only the 5 named classes that the
v2/v3/v4 experiments did not already fetch, taking the vocabulary from
10 to 15.

Layout matches the existing cache so downstream tooling is unchanged:

    data/videos/<split>/<Class>/Signer_NN_*.mov

Splits are assigned by signer, consistent with download_ksl_subset.py.
Existing files are skipped, so the script is safe to re-run.
"""

import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

DATASET = "ekitabu/kenyan-sign-language-videos"

# Kaggle drops connections when single-file requests arrive too fast.
MAX_ATTEMPTS = 5
BASE_BACKOFF = 4.0
REQUEST_SPACING = 1.5

REPO = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    REPO / "backend" / "training" / "data" / "metadata" / "ksl_file_manifest.json"
)
DOWNLOAD_DIR = REPO / "backend" / "training" / "data" / "videos"

# The 5 named classes not already present locally.
NEW_CLASSES = [
    "Apple",
    "Colour",
    "Sweater",
    "Tomatoes",
    "Tortoise",
]

TRAIN_SIGNERS = set(range(1, 11))
VAL_SIGNERS = set(range(11, 13))
TEST_SIGNERS = set(range(13, 16))
ALLOWED = TRAIN_SIGNERS | VAL_SIGNERS | TEST_SIGNERS


def split_for(signer_id):
    if signer_id in TRAIN_SIGNERS:
        return "train"
    if signer_id in VAL_SIGNERS:
        return "val"
    return "test"


def main():
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    selected = []
    for item in manifest:
        name = item.get("name", "")
        if "/" not in name:
            continue

        class_name, filename = name.split("/", 1)
        if class_name not in NEW_CLASSES:
            continue

        match = re.search(r"Signer_(\d+)", filename, re.IGNORECASE)
        if not match:
            continue

        signer_id = int(match.group(1))
        if signer_id not in ALLOWED:
            continue

        selected.append(
            {
                "kaggle_name": name,
                "class_name": class_name,
                "signer_id": signer_id,
                "split": split_for(signer_id),
                "size": item.get("size", 0),
            }
        )

    by_class = Counter(i["class_name"] for i in selected)
    total_gb = sum(i["size"] for i in selected) / (1024**3)

    print("=" * 62)
    print("FADHILI — NAMED VOCABULARY EXPANSION (10 -> 15 classes)")
    print("=" * 62)
    for name in NEW_CLASSES:
        print(f"  {name:<12}{by_class[name]:>4} videos")
    print(f"\n  total {len(selected)} videos, {total_gb:.2f} GB")
    print()

    done = skipped = failed = 0

    for index, item in enumerate(selected, start=1):
        destination = (
            DOWNLOAD_DIR / item["split"] / item["class_name"]
        )
        destination.mkdir(parents=True, exist_ok=True)

        target = destination / Path(item["kaggle_name"]).name

        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue

        # Kaggle throttles rapid single-file requests and drops the
        # connection ("Remote end closed connection without response").
        # Retry with exponential backoff and pace the requests.
        ok = False
        for attempt in range(MAX_ATTEMPTS):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    DATASET,
                    "-f",
                    item["kaggle_name"],
                    "-p",
                    str(destination),
                    "--unzip",
                ],
                capture_output=True,
                text=True,
            )

            if target.exists() and target.stat().st_size > 0:
                ok = True
                break

            time.sleep(BASE_BACKOFF * (2**attempt))

        if ok:
            done += 1
            time.sleep(REQUEST_SPACING)
        else:
            failed += 1
            print(f"  FAIL {item['kaggle_name']}", flush=True)

        if index % 20 == 0 or index == len(selected):
            print(
                f"  [{index}/{len(selected)}] "
                f"downloaded={done} skipped={skipped} failed={failed}",
                flush=True,
            )

    print(
        f"\nDONE  downloaded={done} skipped={skipped} failed={failed}"
    )


if __name__ == "__main__":
    main()
