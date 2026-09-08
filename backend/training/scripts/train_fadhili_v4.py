from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIG
# ============================================================

SEED = 42

SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 150

BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 5e-4

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
    / "features_v4"
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

for directory in [
    MODEL_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    METADATA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD TRAIN / VALIDATION ONLY
# ============================================================

def load_split(split_name):

    X = []
    y = []

    split_root = FEATURE_ROOT / split_name

    print()
    print("=" * 70)
    print(f"LOADING {split_name.upper()}")
    print("=" * 70)

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = split_root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(class_dir)

        files = sorted(class_dir.glob("*.npy"))

        print(
            f"{class_name:<15}: "
            f"{len(files):>3}"
        )

        for path in files:

            sequence = np.load(path).astype(np.float32)

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURES_PER_FRAME,
            ):
                raise ValueError(
                    f"Bad shape: {path} "
                    f"{sequence.shape}"
                )

            if not np.isfinite(sequence).all():
                raise ValueError(
                    f"NaN/Inf: {path}"
                )

            X.append(sequence)
            y.append(class_index)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
    )


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")


print()
print("=" * 70)
print("DATA")
print("=" * 70)
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print()
print("TEST SET IS LOCKED AND WILL NOT BE LOADED.")


# ============================================================
# CLASS WEIGHTS
# ============================================================

counts = np.bincount(
    y_train,
    minlength=len(CLASS_NAMES),
)

class_weights = {}

for i, count in enumerate(counts):

    class_weights[i] = (
        len(y_train)
        / (len(CLASS_NAMES) * count)
    )


# ============================================================
# AUGMENTATION
# ============================================================

def augment_sequence(sequence):
    """
    Training-only augmentation.

    Feature layout:
        pose = first 24 values
        left hand = next 63
        right hand = final 63

    Augmentation intentionally remains mild.
    """

    x = tf.identity(sequence)

    # --------------------------------------------------------
    # 1. Small coordinate noise
    # --------------------------------------------------------

    noise = tf.random.normal(
        tf.shape(x),
        mean=0.0,
        stddev=0.008,
        dtype=tf.float32,
    )

    x = x + noise


    # --------------------------------------------------------
    # 2. Global magnitude / scale perturbation
    # --------------------------------------------------------

    scale = tf.random.uniform(
        [],
        minval=0.92,
        maxval=1.08,
    )

    x = x * scale


    # --------------------------------------------------------
    # 3. Feature dropout
    #
    # Randomly suppress a small number of coordinates.
    # This discourages dependence on individual landmarks.
    # --------------------------------------------------------

    keep_mask = tf.cast(
        tf.random.uniform(
            tf.shape(x)
        ) > 0.025,
        tf.float32,
    )

    x = x * keep_mask


    # --------------------------------------------------------
    # 4. Temporal frame dropout
    #
    # Occasionally replace a frame with the previous frame.
    # Simulates missed / unstable landmark detections.
    # --------------------------------------------------------

    frame_keep = tf.cast(
        tf.random.uniform(
            [SEQUENCE_LENGTH, 1]
        ) > 0.05,
        tf.float32,
    )

    previous = tf.concat(
        [
            x[0:1],
            x[:-1],
        ],
        axis=0,
    )

    x = (
        x * frame_keep
        + previous * (1.0 - frame_keep)
    )


    # --------------------------------------------------------
    # 5. Temporal shift
    #
    # Shift the sequence slightly in time.
    # --------------------------------------------------------

    shift = tf.random.uniform(
        [],
        minval=-2,
        maxval=3,
        dtype=tf.int32,
    )

    x = tf.roll(
        x,
        shift=shift,
        axis=0,
    )

    return x


# ============================================================
# TF DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_dataset = train_dataset.shuffle(
    len(X_train),
    seed=SEED,
    reshuffle_each_iteration=True,
)

train_dataset = train_dataset.map(
    lambda x, y: (
        augment_sequence(x),
        y,
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
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

    # Lightweight temporal convolution
    x = layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.SpatialDropout1D(0.25)(x)


    x = layers.Conv1D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)

    x = layers.BatchNormalization()(x)


    # Smaller recurrent section than v2
    x = layers.Bidirectional(
        layers.GRU(
            48,
            return_sequences=True,
            dropout=0.30,
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
    )(x)


    # Temporal pooling reduces parameters and forces the
    # classifier to reason over the entire gesture.
    average_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)

    x = layers.Concatenate()(
        [
            average_pool,
            max_pool,
        ]
    )


    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(2e-4),
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.45)(x)


    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="sign_probabilities",
    )(x)


    return keras.Model(
        inputs,
        outputs,
        name="Fadhili_KSL_v3",
    )


model = build_model()


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


print()
print("=" * 70)
print("FADHILI V4")
print("=" * 70)

model.summary()

print()
print(
    "MODEL PARAMETERS:",
    f"{model.count_params():,}",
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = (
    CHECKPOINT_DIR
    / "fadhili_v4_best.keras"
)


callbacks = [

    keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),

    keras.callbacks.CSVLogger(
        str(
            LOG_DIR
            / "fadhili_v4_training.csv"
        )
    ),

    keras.callbacks.TerminateOnNaN(),
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("TRAINING FADHILI V4")
print("=" * 70)

history = model.fit(
    train_dataset,

    validation_data=val_dataset,

    epochs=MAX_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1,
)


# ============================================================
# LOAD BEST VALIDATION MODEL
# ============================================================

model = keras.models.load_model(
    checkpoint_path
)


# ============================================================
# VALIDATION
# ============================================================

val_loss, val_accuracy = model.evaluate(
    val_dataset,
    verbose=0,
)


print()
print("=" * 70)
print("V3 VALIDATION RESULT")
print("=" * 70)

print(
    f"Validation loss:     {val_loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{val_accuracy:.2%}"
)


# ============================================================
# SAVE FROZEN CANDIDATE
# ============================================================

candidate_path = (
    MODEL_DIR
    / "fadhili_ksl_v4_candidate.keras"
)

model.save(candidate_path)


# ============================================================
# SAVE RESULTS
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_epoch = int(
    np.argmax(
        history.history["val_accuracy"]
    )
) + 1


results = {

    "model_name":
        "Fadhili_KSL_v3",

    "classes":
        CLASS_NAMES,

    "input_shape": [
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    ],

    "model_parameters":
        int(model.count_params()),

    "train_samples":
        int(len(X_train)),

    "validation_samples":
        int(len(X_val)),

    "test_policy":
        "Test signers 13-15 were not loaded or evaluated.",

    "augmentation": {
        "coordinate_noise_std": 0.008,
        "scale_range": [
            0.92,
            1.08,
        ],
        "coordinate_dropout": 0.025,
        "temporal_frame_dropout": 0.05,
        "temporal_shift_frames": [
            -2,
            2,
        ],
    },

    "epochs_trained":
        len(history.history["loss"]),

    "best_validation_epoch":
        best_val_epoch,

    "best_validation_accuracy":
        float(best_val_accuracy),

    "loaded_candidate_validation_accuracy":
        float(val_accuracy),

    "loaded_candidate_validation_loss":
        float(val_loss),
}


results_path = (
    METADATA_DIR
    / "fadhili_v4_validation_results.json"
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
# FINISH
# ============================================================

print()
print("=" * 70)
print("FADHILI V4 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Epochs trained: "
    f"{len(history.history['loss'])}"
)

print(
    f"Best validation epoch: "
    f"{best_val_epoch}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2%}"
)

print(
    f"Candidate validation accuracy: "
    f"{val_accuracy:.2%}"
)

print()
print("Candidate model:")
print(candidate_path)

print()
print("Validation results:")
print(results_path)

print()
print(
    "TEST SET REMAINS LOCKED."
)


