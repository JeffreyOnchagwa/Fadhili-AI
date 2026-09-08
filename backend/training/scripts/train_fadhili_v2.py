from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 150

BATCH_SIZE = 32
MAX_EPOCHS = 150
LEARNING_RATE = 1e-3

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

MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "models"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "checkpoints"
)

LOG_DIR = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "logs"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "metadata"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split_name):
    """
    Load all .npy feature sequences for one split.

    Expected layout:
        features/
            train/
                Agreement/
                    *.npy
                Friend/
                    *.npy
                ...
    """

    X = []
    y = []
    paths = []

    split_root = FEATURE_ROOT / split_name

    print()
    print("=" * 70)
    print(f"LOADING {split_name.upper()} DATA")
    print("=" * 70)

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = split_root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        files = sorted(class_dir.glob("*.npy"))

        print(
            f"{class_name:<15} "
            f"{len(files):>4} sequences"
        )

        for feature_file in files:

            sequence = np.load(feature_file).astype(np.float32)

            expected_shape = (
                SEQUENCE_LENGTH,
                FEATURES_PER_FRAME,
            )

            if sequence.shape != expected_shape:
                raise ValueError(
                    f"Invalid shape in {feature_file}: "
                    f"{sequence.shape}, expected {expected_shape}"
                )

            if not np.isfinite(sequence).all():
                raise ValueError(
                    f"NaN or Inf detected in {feature_file}"
                )

            X.append(sequence)
            y.append(class_index)
            paths.append(str(feature_file))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print("-" * 70)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y, paths


X_train, y_train, train_paths = load_split("train")
X_val, y_val, val_paths = load_split("val")
X_test, y_test, test_paths = load_split("test")


print()
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Train:      {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")
print(f"Total:      {len(X_train) + len(X_val) + len(X_test)}")


expected_total = 742

actual_total = (
    len(X_train)
    + len(X_val)
    + len(X_test)
)

if actual_total != expected_total:
    raise RuntimeError(
        f"Expected {expected_total} sequences, "
        f"but found {actual_total}"
    )


# ============================================================
# SHUFFLE TRAINING DATA
# ============================================================

indices = np.arange(len(X_train))
np.random.shuffle(indices)

X_train = X_train[indices]
y_train = y_train[indices]


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=len(CLASS_NAMES),
)

total_samples = len(y_train)
num_classes = len(CLASS_NAMES)

class_weights = {}

for class_index, count in enumerate(class_counts):

    if count == 0:
        raise RuntimeError(
            f"No training samples for "
            f"{CLASS_NAMES[class_index]}"
        )

    class_weights[class_index] = (
        total_samples
        / (num_classes * count)
    )


print()
print("=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:<15} "
        f"{class_counts[i]:>4} samples | "
        f"weight={class_weights[i]:.4f}"
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    inputs = keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME,
        ),
        name="landmark_sequence",
    )

    # Temporal feature extraction
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.SpatialDropout1D(0.20)(x)

    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    # Temporal sequence modelling
    x = layers.Bidirectional(
        layers.GRU(
            96,
            return_sequences=True,
            dropout=0.20,
        )
    )(x)

    x = layers.Bidirectional(
        layers.GRU(
            64,
            return_sequences=False,
            dropout=0.20,
        )
    )(x)

    # Classification head
    x = layers.Dense(
        128,
        activation="relu",
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.35)(x)

    x = layers.Dense(
        64,
        activation="relu",
    )(x)

    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="sign_probabilities",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Fadhili_KSL_v2",
    )

    return model


model = build_model()

optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


print()
print("=" * 70)
print("FADHILI AI v2 MODEL")
print("=" * 70)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = (
    CHECKPOINT_DIR
    / "fadhili_v2_best.keras"
)

callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
    ),

    keras.callbacks.CSVLogger(
        str(LOG_DIR / "fadhili_v2_training.csv")
    ),

    keras.callbacks.TerminateOnNaN(),
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val,
    ),

    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=callbacks,

    shuffle=True,

    verbose=1,
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model = keras.models.load_model(
    best_model_path
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0,
)

print()
print(
    f"Validation loss:     {val_loss:.4f}"
)
print(
    f"Validation accuracy: {val_accuracy:.4%}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL UNSEEN-SIGNER TEST")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0,
)

print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4%}")


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=0,
)

predictions = np.argmax(
    probabilities,
    axis=1,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = tf.math.confusion_matrix(
    y_test,
    predictions,
    num_classes=len(CLASS_NAMES),
).numpy()


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(confusion_matrix)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print()
print("=" * 70)
print("PER-CLASS TEST ACCURACY")
print("=" * 70)

per_class_results = {}

for class_index, class_name in enumerate(CLASS_NAMES):

    mask = y_test == class_index

    total = int(np.sum(mask))

    if total == 0:
        accuracy = None
        correct = 0
    else:
        correct = int(
            np.sum(
                predictions[mask]
                == y_test[mask]
            )
        )

        accuracy = correct / total

    per_class_results[class_name] = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
    }

    if accuracy is None:
        print(
            f"{class_name:<15}: "
            f"No test samples"
        )
    else:
        print(
            f"{class_name:<15}: "
            f"{correct:>2}/{total:<2} "
            f"({accuracy:.2%})"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR
    / "fadhili_ksl_v2.keras"
)

model.save(final_model_path)


# ============================================================
# SAVE RESULTS
# ============================================================

history_data = {}

for key, values in history.history.items():
    history_data[key] = [
        float(value)
        for value in values
    ]


results = {

    "model_name": "Fadhili_KSL_v2",

    "classes": CLASS_NAMES,

    "class_to_index": {
        class_name: index
        for index, class_name
        in enumerate(CLASS_NAMES)
    },

    "input_shape": [
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    ],

    "dataset": {
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "total_samples": int(actual_total),
        "split_strategy": (
            "signer-independent: "
            "train=01-10, val=11-12, test=13-15"
        ),
    },

    "validation": {
        "loss": float(val_loss),
        "accuracy": float(val_accuracy),
    },

    "test": {
        "loss": float(test_loss),
        "accuracy": float(test_accuracy),
    },

    "per_class_test_accuracy": per_class_results,

    "confusion_matrix": confusion_matrix.tolist(),

    "training_history": history_data,
}


results_path = (
    METADATA_DIR
    / "fadhili_v2_training_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print(f"Best checkpoint:")
print(best_model_path)

print()
print(f"Final model:")
print(final_model_path)

print()
print(f"Training log:")
print(
    LOG_DIR
    / "fadhili_v2_training.csv"
)

print()
print(f"Results:")
print(results_path)

print()
print(
    f"FINAL UNSEEN-SIGNER TEST ACCURACY: "
    f"{test_accuracy:.2%}"
)

print()
print("=" * 70)