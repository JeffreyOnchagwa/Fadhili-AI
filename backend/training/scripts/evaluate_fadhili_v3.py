from pathlib import Path
import json

import numpy as np
from tensorflow import keras


# ============================================================
# CONFIG
# ============================================================

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 150

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "features"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "models"
    / "fadhili_ksl_v3_candidate.keras"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "metadata"
)


# ============================================================
# LOAD TEST SET
# ============================================================

X_test = []
y_test = []
test_paths = []

print("=" * 70)
print("LOADING LOCKED UNSEEN-SIGNER TEST SET")
print("=" * 70)

for class_index, class_name in enumerate(CLASS_NAMES):

    class_dir = (
        FEATURE_ROOT
        / "test"
        / class_name
    )

    files = sorted(
        class_dir.glob("*.npy")
    )

    print(
        f"{class_name:<15}: "
        f"{len(files):>3}"
    )

    for path in files:

        sequence = np.load(
            path
        ).astype(np.float32)

        expected_shape = (
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME,
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Invalid shape: {path} "
                f"{sequence.shape}"
            )

        if not np.isfinite(sequence).all():
            raise ValueError(
                f"NaN/Inf detected: {path}"
            )

        X_test.append(sequence)
        y_test.append(class_index)
        test_paths.append(str(path))


X_test = np.asarray(
    X_test,
    dtype=np.float32,
)

y_test = np.asarray(
    y_test,
    dtype=np.int32,
)


print()
print("Test shape:", X_test.shape)

if len(X_test) != 150:
    raise RuntimeError(
        f"Expected 150 test sequences, "
        f"found {len(X_test)}"
    )


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING FROZEN FADHILI V3")
print("=" * 70)

model = keras.models.load_model(
    MODEL_PATH
)

print(
    "Parameters:",
    f"{model.count_params():,}",
)


# ============================================================
# EVALUATE
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    batch_size=32,
    verbose=0,
)


probabilities = model.predict(
    X_test,
    batch_size=32,
    verbose=0,
)

predictions = np.argmax(
    probabilities,
    axis=1,
)


print()
print("=" * 70)
print("FADHILI V3 — UNSEEN-SIGNER TEST")
print("=" * 70)

print(
    f"Test loss:     "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.2%}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

num_classes = len(CLASS_NAMES)

confusion_matrix = np.zeros(
    (num_classes, num_classes),
    dtype=np.int32,
)

for true_label, predicted_label in zip(
    y_test,
    predictions,
):

    confusion_matrix[
        true_label,
        predicted_label
    ] += 1


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(confusion_matrix)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print("=" * 70)
print("PER-CLASS TEST ACCURACY")
print("=" * 70)

per_class = {}

for class_index, class_name in enumerate(CLASS_NAMES):

    mask = (
        y_test == class_index
    )

    total = int(
        np.sum(mask)
    )

    correct = int(
        np.sum(
            predictions[mask]
            == y_test[mask]
        )
    )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    per_class[class_name] = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }

    print(
        f"{class_name:<15}: "
        f"{correct:>2}/{total:<2} "
        f"({accuracy:.2%})"
    )


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

confidence = np.max(
    probabilities,
    axis=1,
)

correct_mask = (
    predictions == y_test
)

incorrect_mask = (
    predictions != y_test
)


mean_confidence = float(
    np.mean(confidence)
)

mean_correct_confidence = float(
    np.mean(
        confidence[correct_mask]
    )
)

if np.any(incorrect_mask):

    mean_incorrect_confidence = float(
        np.mean(
            confidence[incorrect_mask]
        )
    )

else:

    mean_incorrect_confidence = 0.0


print()
print("=" * 70)
print("CONFIDENCE")
print("=" * 70)

print(
    f"Overall mean confidence:   "
    f"{mean_confidence:.2%}"
)

print(
    f"Correct predictions:       "
    f"{mean_correct_confidence:.2%}"
)

print(
    f"Incorrect predictions:     "
    f"{mean_incorrect_confidence:.2%}"
)


# ============================================================
# COMPARE AGAINST V2
# ============================================================

V2_ACCURACY = 0.47333332896232605

absolute_improvement = (
    float(test_accuracy)
    - V2_ACCURACY
)

relative_error_v2 = (
    1.0 - V2_ACCURACY
)

relative_error_v3 = (
    1.0 - float(test_accuracy)
)

error_reduction = (
    (
        relative_error_v2
        - relative_error_v3
    )
    / relative_error_v2
)


print()
print("=" * 70)
print("V2 → V3 COMPARISON")
print("=" * 70)

print(
    f"V2 accuracy: "
    f"{V2_ACCURACY:.2%}"
)

print(
    f"V3 accuracy: "
    f"{test_accuracy:.2%}"
)

print(
    f"Absolute change: "
    f"{absolute_improvement:+.2%}"
)

print(
    f"Relative error reduction: "
    f"{error_reduction:+.2%}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model":
        "Fadhili_KSL_v3",

    "evaluation":
        "Signer-independent locked test",

    "test_signers":
        "13-15",

    "samples":
        int(len(X_test)),

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(test_accuracy),

    "v2_test_accuracy":
        float(V2_ACCURACY),

    "absolute_improvement":
        float(absolute_improvement),

    "relative_error_reduction":
        float(error_reduction),

    "confusion_matrix":
        confusion_matrix.tolist(),

    "per_class_accuracy":
        per_class,

    "confidence": {

        "mean":
            mean_confidence,

        "correct":
            mean_correct_confidence,

        "incorrect":
            mean_incorrect_confidence,
    },
}


output_path = (
    METADATA_DIR
    / "fadhili_v3_test_results.json"
)


with open(
    output_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=2,
    )


print()
print("=" * 70)
print("FINAL RESULT SAVED")
print("=" * 70)

print(output_path)