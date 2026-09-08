from pathlib import Path
from collections import Counter, defaultdict
import json
import random
import re
import gc

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
MAX_EPOCHS = 80
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

FOLDS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
]


PROJECT_ROOT = Path(__file__).resolve().parents[3]

METADATA_PATH = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "metadata"
    / "fadhili_v2_features.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "checkpoints"
    / "cv_v3"
)

LOG_DIR = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "logs"
    / "cv_v3"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "data"
    / "metadata"
    / "fadhili_v3_cross_validation.json"
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# METADATA
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)


def extract_signer_id(video_name):

    match = re.search(
        r"Signer_(\d+)",
        video_name,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Cannot extract signer from {video_name}"
        )

    return int(match.group(1))


development_records = []

for record in metadata["records"]:

    signer_id = extract_signer_id(
        record["video"]
    )

    if 1 <= signer_id <= 12:

        development_records.append(
            {
                **record,
                "signer_id": signer_id,
            }
        )


if len(development_records) != 592:
    raise RuntimeError(
        f"Expected 592 development records, "
        f"found {len(development_records)}"
    )


# ============================================================
# LOAD ONE RECORD
# ============================================================

def load_feature(record):

    path = (
        PROJECT_ROOT
        / Path(record["feature_file"])
    )

    sequence = np.load(
        path
    ).astype(np.float32)

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURES_PER_FRAME,
    ):
        raise ValueError(
            f"Invalid shape: {path}"
        )

    if not np.isfinite(sequence).all():
        raise ValueError(
            f"NaN/Inf: {path}"
        )

    return sequence


# ============================================================
# BUILD DATA ARRAYS
# ============================================================

def build_arrays(records):

    X = []
    y = []

    for record in records:

        X.append(
            load_feature(record)
        )

        y.append(
            CLASS_NAMES.index(
                record["class"]
            )
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=np.int32,
        ),
    )


# ============================================================
# AUGMENTATION
# ============================================================

def augment_sequence(sequence):

    x = tf.identity(sequence)

    # Mild coordinate noise
    x = x + tf.random.normal(
        tf.shape(x),
        mean=0.0,
        stddev=0.008,
    )

    # Scale perturbation
    scale = tf.random.uniform(
        [],
        minval=0.92,
        maxval=1.08,
    )

    x = x * scale

    # Coordinate dropout
    keep_mask = tf.cast(
        tf.random.uniform(
            tf.shape(x)
        ) > 0.025,
        tf.float32,
    )

    x = x * keep_mask

    # Frame dropout
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
        + previous
        * (1.0 - frame_keep)
    )

    # Small temporal shift
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
# MODEL
# ============================================================

def build_model():

    inputs = keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURES_PER_FRAME,
        )
    )

    x = layers.Conv1D(
        64,
        5,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(
            1e-4
        ),
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.SpatialDropout1D(
        0.25
    )(x)

    x = layers.Conv1D(
        64,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(
            1e-4
        ),
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(
        layers.GRU(
            48,
            return_sequences=True,
            dropout=0.30,
            kernel_regularizer=keras.regularizers.l2(
                1e-4
            ),
        )
    )(x)

    average_pool = (
        layers.GlobalAveragePooling1D()(x)
    )

    max_pool = (
        layers.GlobalMaxPooling1D()(x)
    )

    x = layers.Concatenate()(
        [
            average_pool,
            max_pool,
        ]
    )

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(
            2e-4
        ),
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.45
    )(x)

    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
    )(x)

    return keras.Model(
        inputs,
        outputs,
        name="Fadhili_KSL_CV",
    )


# ============================================================
# RESULTS
# ============================================================

fold_results = []

combined_true = []
combined_pred = []


# ============================================================
# CROSS-VALIDATION
# ============================================================

for fold_index, validation_signers in enumerate(
    FOLDS,
    start=1,
):

    print()
    print("=" * 70)
    print(
        f"FOLD {fold_index}/6"
    )
    print("=" * 70)

    print(
        "Validation signers:",
        validation_signers,
    )

    train_records = [
        record
        for record in development_records
        if record["signer_id"]
        not in validation_signers
    ]

    val_records = [
        record
        for record in development_records
        if record["signer_id"]
        in validation_signers
    ]

    X_train, y_train = build_arrays(
        train_records
    )

    X_val, y_val = build_arrays(
        val_records
    )

    print(
        "Train:",
        X_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
    )


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    counts = np.bincount(
        y_train,
        minlength=len(CLASS_NAMES),
    )

    class_weights = {}

    for i, count in enumerate(counts):

        if count == 0:
            raise RuntimeError(
                f"Fold {fold_index}: "
                f"zero training samples for "
                f"{CLASS_NAMES[i]}"
            )

        class_weights[i] = (
            len(y_train)
            / (
                len(CLASS_NAMES)
                * count
            )
        )


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (X_train, y_train)
        )
    )

    train_dataset = (
        train_dataset
        .shuffle(
            len(X_train),
            seed=SEED + fold_index,
            reshuffle_each_iteration=True,
        )
        .map(
            lambda x, y: (
                augment_sequence(x),
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


    val_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (X_val, y_val)
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


    # --------------------------------------------------------
    # FRESH MODEL
    # --------------------------------------------------------

    keras.backend.clear_session()

    random.seed(
        SEED + fold_index
    )

    np.random.seed(
        SEED + fold_index
    )

    tf.random.set_seed(
        SEED + fold_index
    )

    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=["accuracy"],
    )


    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"fold_{fold_index}_best.keras"
    )

    log_path = (
        LOG_DIR
        / f"fold_{fold_index}.csv"
    )

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=12,
            restore_best_weights=True,
            verbose=0,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0,
        ),

        keras.callbacks.CSVLogger(
            str(log_path)
        ),

        keras.callbacks.TerminateOnNaN(),
    ]


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )


    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    model = keras.models.load_model(
        checkpoint_path
    )


    val_loss, val_accuracy = (
        model.evaluate(
            val_dataset,
            verbose=0,
        )
    )


    probabilities = model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )


    # --------------------------------------------------------
    # PER-CLASS
    # --------------------------------------------------------

    per_class = {}

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        mask = (
            y_val == class_index
        )

        total = int(
            np.sum(mask)
        )

        correct = int(
            np.sum(
                predictions[mask]
                == y_val[mask]
            )
        )

        accuracy = (
            correct / total
            if total
            else None
        )

        per_class[class_name] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }


    best_epoch = int(
        np.argmax(
            history.history[
                "val_accuracy"
            ]
        )
    ) + 1


    result = {

        "fold":
            fold_index,

        "validation_signers":
            list(validation_signers),

        "train_samples":
            int(len(X_train)),

        "validation_samples":
            int(len(X_val)),

        "epochs_trained":
            len(history.history["loss"]),

        "best_epoch":
            best_epoch,

        "validation_loss":
            float(val_loss),

        "validation_accuracy":
            float(val_accuracy),

        "per_class":
            per_class,
    }


    fold_results.append(
        result
    )

    combined_true.extend(
        y_val.tolist()
    )

    combined_pred.extend(
        predictions.tolist()
    )


    print()
    print(
        f"FOLD {fold_index} RESULT:"
    )

    print(
        f"Accuracy: "
        f"{val_accuracy:.2%}"
    )

    print(
        f"Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )


    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    del model
    del X_train
    del y_train
    del X_val
    del y_val

    gc.collect()

    keras.backend.clear_session()


# ============================================================
# AGGREGATE RESULTS
# ============================================================

accuracies = np.asarray(
    [
        result[
            "validation_accuracy"
        ]
        for result in fold_results
    ],
    dtype=np.float32,
)


mean_accuracy = float(
    np.mean(accuracies)
)

std_accuracy = float(
    np.std(
        accuracies,
        ddof=1,
    )
)

minimum_accuracy = float(
    np.min(accuracies)
)

maximum_accuracy = float(
    np.max(accuracies)
)


combined_true = np.asarray(
    combined_true,
    dtype=np.int32,
)

combined_pred = np.asarray(
    combined_pred,
    dtype=np.int32,
)


overall_accuracy = float(
    np.mean(
        combined_true
        == combined_pred
    )
)


# ============================================================
# AGGREGATED PER-CLASS ACCURACY
# ============================================================

aggregate_per_class = {}


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    mask = (
        combined_true
        == class_index
    )

    total = int(
        np.sum(mask)
    )

    correct = int(
        np.sum(
            combined_pred[mask]
            == combined_true[mask]
        )
    )

    accuracy = (
        correct / total
        if total
        else None
    )

    aggregate_per_class[
        class_name
    ] = {

        "correct":
            correct,

        "total":
            total,

        "accuracy":
            accuracy,
    }


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("CROSS-VALIDATION COMPLETE")
print("=" * 70)


for result in fold_results:

    signers = ", ".join(
        f"{signer:02d}"
        for signer
        in result[
            "validation_signers"
        ]
    )

    print(
        f"Fold {result['fold']} "
        f"(signers {signers}): "
        f"{result['validation_accuracy']:.2%}"
    )


print()
print(
    f"Mean fold accuracy: "
    f"{mean_accuracy:.2%}"
)

print(
    f"Standard deviation: "
    f"{std_accuracy:.2%}"
)

print(
    f"Worst fold: "
    f"{minimum_accuracy:.2%}"
)

print(
    f"Best fold: "
    f"{maximum_accuracy:.2%}"
)

print(
    f"Combined out-of-signer accuracy: "
    f"{overall_accuracy:.2%}"
)


print()
print("=" * 70)
print("AGGREGATED PER-CLASS ACCURACY")
print("=" * 70)


for class_name in CLASS_NAMES:

    result = aggregate_per_class[
        class_name
    ]

    print(
        f"{class_name:<15}: "
        f"{result['correct']:>3}/"
        f"{result['total']:<3} "
        f"({result['accuracy']:.2%})"
    )


# ============================================================
# SAVE
# ============================================================

output = {

    "experiment":
        "Fadhili v3 signer-group cross-validation",

    "development_signers":
        list(range(1, 13)),

    "locked_test_signers":
        [13, 14, 15],

    "locked_test_used":
        False,

    "folds":
        fold_results,

    "summary": {

        "mean_fold_accuracy":
            mean_accuracy,

        "std_fold_accuracy":
            std_accuracy,

        "minimum_fold_accuracy":
            minimum_accuracy,

        "maximum_fold_accuracy":
            maximum_accuracy,

        "combined_out_of_signer_accuracy":
            overall_accuracy,
    },

    "aggregate_per_class":
        aggregate_per_class,
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        output,
        file,
        indent=2,
    )


print()
print("Results saved:")
print(RESULTS_PATH)

print()
print(
    "Signers 13-15 were NOT loaded "
    "during this experiment."
)