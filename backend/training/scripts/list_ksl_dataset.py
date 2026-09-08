import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "ekitabu/kenyan-sign-language-videos"

# Kaggle CLI maximum page size is 200
PAGE_SIZE = 200

# Number of times to retry a failed Kaggle request
MAX_RETRIES = 5

OUTPUT_DIR = Path("backend/training/data/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STORAGE
# ============================================================

all_files = []

page_token = None
page = 1


# ============================================================
# DOWNLOAD COMPLETE FILE MANIFEST
# ============================================================

while True:

    print(f"Fetching page {page}...")

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "files",
        DATASET,
        "--page-size",
        str(PAGE_SIZE),
        "--format",
        "json",
    ]

    if page_token:
        cmd.extend(
            [
                "--page-token",
                page_token,
            ]
        )

    # --------------------------------------------------------
    # Retry logic
    # --------------------------------------------------------

    result = None

    for attempt in range(1, MAX_RETRIES + 1):

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            break

        print(
            f"  Request failed "
            f"(attempt {attempt}/{MAX_RETRIES})"
        )

        if result.stderr.strip():
            print(f"  {result.stderr.strip()}")

        if attempt == MAX_RETRIES:
            print()
            print(
                f"Kaggle failed after "
                f"{MAX_RETRIES} attempts."
            )
            sys.exit(1)

        # Increasing delay:
        # 5 sec, 10 sec, 15 sec, 20 sec...
        wait_seconds = attempt * 5

        print(
            f"  Waiting {wait_seconds} seconds "
            f"before retrying..."
        )

        time.sleep(wait_seconds)

    # --------------------------------------------------------
    # Read Kaggle output
    # --------------------------------------------------------

    output = result.stdout.strip()

    if not output:
        print("Kaggle returned an empty response.")
        sys.exit(1)

    # Kaggle CLI outputs:
    #
    # Next Page Token = XXXXX
    # [
    #   {...},
    #   {...}
    # ]
    #
    # We therefore separate the token from the JSON.

    token_match = re.match(
        r"Next Page Token = (.+?)\r?\n",
        output,
    )

    if token_match:

        next_page_token = token_match.group(1).strip()

        json_text = output[
            token_match.end():
        ].strip()

    else:

        next_page_token = None

        # Find the beginning of the JSON array.
        json_start = output.find("[")

        if json_start == -1:
            print()
            print(
                "Could not find JSON in "
                "Kaggle response."
            )
            print()
            print(output[:1000])
            sys.exit(1)

        json_text = output[json_start:]

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        files = json.loads(json_text)

    except json.JSONDecodeError as exc:

        print()
        print(
            f"Could not parse page {page}: "
            f"{exc}"
        )

        print()
        print("Beginning of Kaggle response:")
        print(json_text[:1000])

        sys.exit(1)

    # --------------------------------------------------------
    # Store files
    # --------------------------------------------------------

    all_files.extend(files)

    print(
        f"  Files on page: {len(files)} | "
        f"Total collected: {len(all_files)}"
    )

    # --------------------------------------------------------
    # Check whether another page exists
    # --------------------------------------------------------

    if not next_page_token:
        break

    page_token = next_page_token
    page += 1

    # Small delay to avoid hammering Kaggle's API
    time.sleep(1)


# ============================================================
# ANALYSE DATASET
# ============================================================

print()
print("Analysing dataset...")

class_counts = Counter()

class_signers = defaultdict(set)

class_sizes = Counter()

all_signers = set()


for file_info in all_files:

    name = file_info.get("name", "")

    size = file_info.get("size", 0)

    # Ignore files that aren't inside a class folder.
    if "/" not in name:
        continue

    class_name, filename = name.split("/", 1)

    class_counts[class_name] += 1

    class_sizes[class_name] += size

    # Example filename:
    #
    # Signer_01_1.mov

    signer_match = re.search(
        r"Signer_(\d+)",
        filename,
        re.IGNORECASE,
    )

    if signer_match:

        signer_id = signer_match.group(1)

        class_signers[class_name].add(
            signer_id
        )

        all_signers.add(
            signer_id
        )


classes = sorted(
    class_counts.keys(),
    key=str.lower,
)


# ============================================================
# SAVE COMPLETE RAW MANIFEST
# ============================================================

manifest_path = (
    OUTPUT_DIR /
    "ksl_file_manifest.json"
)

with manifest_path.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_files,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

classes_path = (
    OUTPUT_DIR /
    "ksl_classes.txt"
)

with classes_path.open(
    "w",
    encoding="utf-8",
) as f:

    for class_name in classes:

        f.write(
            class_name + "\n"
        )


# ============================================================
# CREATE CLASS SUMMARY
# ============================================================

summary = []

for class_name in classes:

    summary.append(
        {
            "class": class_name,

            "videos":
                class_counts[class_name],

            "signers":
                len(
                    class_signers[
                        class_name
                    ]
                ),

            "size_mb":
                round(
                    class_sizes[
                        class_name
                    ]
                    / (1024 * 1024),
                    2,
                ),
        }
    )


summary_path = (
    OUTPUT_DIR /
    "ksl_class_summary.json"
)

with summary_path.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# CALCULATE TOTAL DATASET SIZE
# ============================================================

total_bytes = sum(
    file_info.get("size", 0)
    for file_info in all_files
)

total_gb = (
    total_bytes /
    (1024 ** 3)
)


# ============================================================
# PRINT DATASET SUMMARY
# ============================================================

print()
print("=" * 70)

print(
    "KSL DATASET SUMMARY"
)

print("=" * 70)

print(
    f"Total files:   "
    f"{len(all_files)}"
)

print(
    f"Total classes: "
    f"{len(classes)}"
)

print(
    f"Total signers: "
    f"{len(all_signers)}"
)

print(
    f"Total size:    "
    f"{total_gb:.2f} GB"
)


# ============================================================
# PRINT CLASSES
# ============================================================

print()
print("CLASSES")

print("-" * 70)

for class_name in classes:

    video_count = (
        class_counts[class_name]
    )

    signer_count = len(
        class_signers[
            class_name
        ]
    )

    size_mb = (
        class_sizes[class_name]
        / (1024 * 1024)
    )

    print(
        f"{class_name:<30} "
        f"{video_count:>5} videos | "
        f"{signer_count:>3} signers | "
        f"{size_mb:>8.2f} MB"
    )


# ============================================================
# PRINT SAVED FILES
# ============================================================

print()
print("=" * 70)

print("FILES SAVED")

print("=" * 70)

print(
    f"Manifest: "
    f"{manifest_path}"
)

print(
    f"Classes:  "
    f"{classes_path}"
)

print(
    f"Summary:  "
    f"{summary_path}"
)

print()
print(
    "Dataset inspection completed successfully."
)