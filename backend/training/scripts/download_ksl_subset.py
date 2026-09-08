import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "ekitabu/kenyan-sign-language-videos"

MANIFEST_PATH = Path(
    "backend/training/data/metadata/ksl_file_manifest.json"
)

DOWNLOAD_DIR = Path(
    "backend/training/data/videos"
)

# Fadhili AI v2 initial vocabulary
SELECTED_CLASSES = [
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

# Signer-independent split
#
# Train:      Signers 01-10
# Validation: Signers 11-12
# Test:       Signers 13-15
#
# The test signers will NEVER be used during training.

TRAIN_SIGNERS = set(range(1, 11))
VAL_SIGNERS = set(range(11, 13))
TEST_SIGNERS = set(range(13, 16))

ALLOWED_SIGNERS = (
    TRAIN_SIGNERS |
    VAL_SIGNERS |
    TEST_SIGNERS
)


# ============================================================
# LOAD MANIFEST
# ============================================================

if not MANIFEST_PATH.exists():
    print(
        f"ERROR: Manifest not found:\n"
        f"{MANIFEST_PATH}"
    )
    sys.exit(1)

with MANIFEST_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    manifest = json.load(f)


# ============================================================
# SELECT FILES
# ============================================================

selected_files = []

for item in manifest:

    name = item.get("name", "")

    if "/" not in name:
        continue

    class_name, filename = name.split("/", 1)

    if class_name not in SELECTED_CLASSES:
        continue

    signer_match = re.search(
        r"Signer_(\d+)",
        filename,
        re.IGNORECASE,
    )

    if not signer_match:
        continue

    signer_id = int(
        signer_match.group(1)
    )

    if signer_id not in ALLOWED_SIGNERS:
        continue

    selected_files.append(
        {
            **item,
            "class_name": class_name,
            "signer_id": signer_id,
        }
    )


# ============================================================
# ASSIGN SPLITS
# ============================================================

def get_split(signer_id):

    if signer_id in TRAIN_SIGNERS:
        return "train"

    if signer_id in VAL_SIGNERS:
        return "val"

    if signer_id in TEST_SIGNERS:
        return "test"

    raise ValueError(
        f"Unknown signer: {signer_id}"
    )


for item in selected_files:
    item["split"] = get_split(
        item["signer_id"]
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

total_bytes = sum(
    item.get("size", 0)
    for item in selected_files
)

total_gb = total_bytes / (1024 ** 3)

class_counts = Counter(
    item["class_name"]
    for item in selected_files
)

split_counts = Counter(
    item["split"]
    for item in selected_files
)

split_sizes = Counter()

for item in selected_files:
    split_sizes[item["split"]] += (
        item.get("size", 0)
    )


# ============================================================
# DISPLAY PLAN
# ============================================================

print()
print("=" * 70)
print("FADHILI AI v2 — SIGNER-INDEPENDENT DATASET")
print("=" * 70)

print(
    f"Classes:      "
    f"{len(SELECTED_CLASSES)}"
)

print(
    f"Total videos: "
    f"{len(selected_files)}"
)

print(
    f"Total size:   "
    f"{total_gb:.2f} GB"
)

print()

print("SIGNER SPLIT")
print("-" * 70)

print(
    "Train:      Signers 01-10"
)

print(
    "Validation: Signers 11-12"
)

print(
    "Test:       Signers 13-15"
)

print()

print("VIDEOS BY SPLIT")
print("-" * 70)

for split in [
    "train",
    "val",
    "test",
]:

    count = split_counts[split]

    size_gb = (
        split_sizes[split]
        / (1024 ** 3)
    )

    print(
        f"{split:<12}"
        f"{count:>5} videos | "
        f"{size_gb:.2f} GB"
    )


print()

print("VIDEOS BY CLASS")
print("-" * 70)

for class_name in SELECTED_CLASSES:

    print(
        f"{class_name:<20}"
        f"{class_counts[class_name]:>4} videos"
    )


# ============================================================
# SAVE SPLIT MANIFEST
# ============================================================

split_manifest_path = Path(
    "backend/training/data/metadata/"
    "fadhili_v2_split.json"
)

with split_manifest_path.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        selected_files,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print(
    f"Split manifest saved to:\n"
    f"{split_manifest_path}"
)


# ============================================================
# CONFIRM DOWNLOAD
# ============================================================

print()
print(
    "Nothing has been downloaded yet."
)

answer = input(
    "\nType DOWNLOAD to begin, "
    "or press Enter to cancel: "
).strip()


if answer != "DOWNLOAD":

    print("Download cancelled.")

    sys.exit(0)


# ============================================================
# DOWNLOAD
# ============================================================

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

total = len(selected_files)


for index, item in enumerate(
    selected_files,
    start=1,
):

    kaggle_filename = item["name"]

    class_name = item["class_name"]

    signer_id = item["signer_id"]

    split = item["split"]

    original_filename = Path(
        kaggle_filename
    ).name

    destination_directory = (
        DOWNLOAD_DIR /
        split /
        class_name
    )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_file = (
        destination_directory /
        original_filename
    )

    # Resume support
    if destination_file.exists():

        print(
            f"[{index}/{total}] "
            f"SKIP: {split}/"
            f"{class_name}/"
            f"{original_filename}"
        )

        continue

    print()
    print(
        f"[{index}/{total}] "
        f"{split.upper()} | "
        f"{class_name} | "
        f"Signer {signer_id:02d}"
    )

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-f",
        kaggle_filename,
        "-p",
        str(destination_directory),
        "--unzip",
    ]

    result = subprocess.run(
        cmd
    )

    if result.returncode != 0:

        print(
            f"WARNING: Failed to download "
            f"{kaggle_filename}"
        )

        print(
            "The script will continue."
        )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(
    f"Dataset location:\n"
    f"{DOWNLOAD_DIR}"
)

print()

print(
    "Train signers:      01-10"
)

print(
    "Validation signers: 11-12"
)

print(
    "Test signers:       13-15"
)

print()

print(
    "The test signers must remain unseen "
    "during model training."
)