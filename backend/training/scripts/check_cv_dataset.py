from pathlib import Path
from collections import Counter, defaultdict
import json
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

METADATA_PATH = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "metadata"
    / "fadhili_v2_features.json"
)


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

records = metadata["records"]


print("=" * 70)
print("FADHILI CROSS-SIGNER DATASET CHECK")
print("=" * 70)

print()
print("Total metadata records:", len(records))


# ============================================================
# EXTRACT SIGNER ID
# ============================================================

def extract_signer_id(video_name):

    match = re.search(
        r"Signer_(\d+)",
        video_name,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Could not extract signer from: {video_name}"
        )

    return int(match.group(1))


# ============================================================
# ANALYZE RECORDS
# ============================================================

signer_counts = Counter()
class_counts = Counter()

signer_class_counts = defaultdict(Counter)

development_records = []
locked_test_records = []


for record in records:

    signer_id = extract_signer_id(
        record["video"]
    )

    class_name = record["class"]

    signer_counts[signer_id] += 1
    class_counts[class_name] += 1

    signer_class_counts[
        signer_id
    ][class_name] += 1

    if 1 <= signer_id <= 12:
        development_records.append(record)

    elif 13 <= signer_id <= 15:
        locked_test_records.append(record)

    else:
        raise ValueError(
            f"Unexpected signer ID: {signer_id}"
        )


# ============================================================
# BASIC SUMMARY
# ============================================================

print()
print("=" * 70)
print("SIGNER COUNTS")
print("=" * 70)

for signer_id in sorted(signer_counts):

    print(
        f"Signer {signer_id:02d}: "
        f"{signer_counts[signer_id]} sequences"
    )


print()
print("=" * 70)
print("CLASS COUNTS")
print("=" * 70)

for class_name in sorted(class_counts):

    print(
        f"{class_name:<15}: "
        f"{class_counts[class_name]}"
    )


print()
print("=" * 70)
print("DEVELOPMENT / LOCKED TEST")
print("=" * 70)

print(
    "Development signers 01-12:",
    len(development_records),
)

print(
    "Locked signers 13-15:",
    len(locked_test_records),
)


# ============================================================
# SIGNER × CLASS MATRIX
# ============================================================

classes = metadata["classes"]

print()
print("=" * 70)
print("SIGNER × CLASS COUNTS")
print("=" * 70)

header = "Signer".ljust(10)

for class_name in classes:
    header += class_name[:7].rjust(9)

print(header)


for signer_id in range(1, 16):

    row = f"{signer_id:02d}".ljust(10)

    for class_name in classes:

        count = signer_class_counts[
            signer_id
        ][class_name]

        row += str(count).rjust(9)

    print(row)


# ============================================================
# VERIFY FEATURE FILES
# ============================================================

missing_features = []

for record in development_records:

    feature_path = (
        PROJECT_ROOT
        / Path(record["feature_file"])
    )

    if not feature_path.exists():

        missing_features.append(
            str(feature_path)
        )


print()
print("=" * 70)
print("FEATURE FILE CHECK")
print("=" * 70)

print(
    "Development feature files:",
    len(development_records),
)

print(
    "Missing feature files:",
    len(missing_features),
)


if missing_features:

    print()

    for path in missing_features:
        print("MISSING:", path)


# ============================================================
# DEFINE CV FOLDS
# ============================================================

folds = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
]


print()
print("=" * 70)
print("CROSS-VALIDATION FOLDS")
print("=" * 70)


for fold_number, validation_signers in enumerate(
    folds,
    start=1,
):

    training_signers = [
        signer
        for signer in range(1, 13)
        if signer not in validation_signers
    ]

    validation_count = sum(
        signer_counts[signer]
        for signer in validation_signers
    )

    training_count = sum(
        signer_counts[signer]
        for signer in training_signers
    )

    print()
    print(
        f"Fold {fold_number}"
    )

    print(
        "  Validation:",
        ", ".join(
            f"{s:02d}"
            for s in validation_signers
        ),
    )

    print(
        "  Training:  ",
        ", ".join(
            f"{s:02d}"
            for s in training_signers
        ),
    )

    print(
        f"  Training sequences:   "
        f"{training_count}"
    )

    print(
        f"  Validation sequences: "
        f"{validation_count}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)

if len(records) != 742:

    print(
        "ERROR: Expected 742 total records."
    )

elif len(development_records) != 592:

    print(
        "WARNING: Development count is not 592."
    )

elif len(locked_test_records) != 150:

    print(
        "ERROR: Locked test count is not 150."
    )

elif missing_features:

    print(
        "ERROR: Some development feature files are missing."
    )

else:

    print(
        "SUCCESS: Cross-validation dataset is ready."
    )

    print(
        "Signers 01-12 may be used for development."
    )

    print(
        "Signers 13-15 remain locked."
    )

print("=" * 70)