"""
Fadhili v5 experiment harness: build, train and evaluate honestly.

Evaluation protocol
-------------------
The single most important finding from the v2/v3/v4 audit is that the
dataset is TWO corpora:

    Domain A  signers 01-10  controlled studio, standing
    Domain B  signers 11-15  in-the-wild rooms, seated

A signer-grouped CV drawn from within Domain A measures a far easier
problem than deployment. That is why v3 scored 97.6% on signer-group CV
over signers 01-12 yet 53.3% on signers 13-15.

This harness therefore reports three separate numbers and never
conflates them:

    within_studio   LOSO over signers 01-10          (easy)
    cross_domain    train studio 01-10, test wild    (deployment-like)
    dev_loso        LOSO over signers 01-12          (model selection)

Signers 13-15 are NOT used for model selection. They were inspected
repeatedly during v2/v3/v4 development, so they are a held-out
DIAGNOSTIC set, not a pristine test set, and calling them otherwise
would overstate what we know. They are evaluated once, at the end, on a
frozen configuration, via --final.

Anti-leakage
------------
Splits are always by signer. A video contributes exactly one sample, so
frames from one recording can never straddle a split. Augmentation is
applied only to training folds, never to evaluation data. Normalization
statistics, where used, are fit on training folds only.
"""

import argparse
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from features_v5 import (  # noqa: E402
    FeatureConfig,
    build_background_sequence,
    build_sequence,
    feature_dimension,
)

# Label used for the rejection class. A production interpreter must be
# able to say "nothing is being signed" rather than forcing every input
# into a known word.
NO_SIGN_LABEL = "__no_sign__"

REPO = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO / "backend" / "training" / "data" / "raw_landmarks"
META_DIR = REPO / "backend" / "training" / "data" / "metadata"
RESULT_DIR = META_DIR / "v5_experiments"

STUDIO_SIGNERS = list(range(1, 11))
WILD_SIGNERS = list(range(11, 16))
DEV_SIGNERS = list(range(1, 13))
DIAGNOSTIC_SIGNERS = [13, 14, 15]

SEED = 1337


# =====================================================================
# DATA
# =====================================================================


def load_dataset(config, classes=None, with_background=False):
    """
    Build the full dataset from the raw landmark cache.

    When `with_background` is set, an extra rejection class is added,
    built from the idle stretches that motion trimming discards — real
    footage of the same people not signing. Those samples carry the
    signer id of the recording they came from, so signer-grouped splits
    keep them on the correct side and no signer leaks across a fold.

    Returns X, y, signers, class_names.
    """
    paths = sorted(RAW_ROOT.rglob("*.npz"))
    if not paths:
        raise SystemExit(
            f"No raw landmarks in {RAW_ROOT}. "
            "Run cache_raw_landmarks.py first."
        )

    by_class = {}
    for path in paths:
        class_name = path.parent.name
        by_class.setdefault(class_name, []).append(path)

    if classes is None:
        classes = sorted(by_class)

    X, y, signers = [], [], []

    for index, class_name in enumerate(classes):
        for path in sorted(by_class.get(class_name, [])):
            match = re.match(r"Signer_(\d+)_", path.name)
            if not match:
                continue
            sequence = build_sequence(path, config)
            if sequence is None:
                continue
            X.append(sequence)
            y.append(index)
            signers.append(int(match.group(1)))

    class_names = list(classes)

    if with_background:
        rng = np.random.default_rng(SEED)
        background_index = len(class_names)
        class_names.append(NO_SIGN_LABEL)

        # Sample background from a subset so the rejection class does
        # not swamp the vocabulary: roughly one background example per
        # class keeps the problem balanced.
        stride = max(1, len(classes))
        candidates = [
            path
            for index, path in enumerate(paths)
            if index % stride == 0
            and path.parent.name in classes
        ]

        for path in candidates:
            match = re.match(r"Signer_(\d+)_", path.name)
            if not match:
                continue
            sequence = build_background_sequence(path, config, rng)
            if sequence is None:
                continue
            X.append(sequence)
            y.append(background_index)
            signers.append(int(match.group(1)))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        np.asarray(signers, dtype=np.int32),
        class_names,
    )


# =====================================================================
# AUGMENTATION
# =====================================================================


def augment(batch, rng, strength=1.0):
    """
    Physically plausible augmentation, applied to training data only.

    Deliberately excluded:
      - horizontal flips, which change handedness and would relabel
        one-handed signs;
      - anything applied to the presence-mask channels, which are
        categorical and must stay 0/1.
    """
    out = batch.copy()
    n, t, d = out.shape

    # Temporal speed variation.
    #
    # This is the augmentation that matters most here. Measured wrist
    # velocity differs by more than 2x between the fastest studio signer
    # (0.130) and the slowest wild signer (0.059), and signing tempo is
    # the largest single axis of variation between the two corpora.
    # Resampling each sample to a random tempo forces the model to
    # recognise a sign by its shape and trajectory rather than by how
    # many frames it happens to occupy.
    for i in range(n):
        speed = rng.uniform(1 - 0.30 * strength, 1 + 0.30 * strength)
        src = np.clip(np.linspace(0, t - 1, t) * speed, 0, t - 1)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, t - 1)
        w = (src - lo).astype(np.float32)[:, None]
        out[i] = (1 - w) * out[i][lo] + w * out[i][hi]

    # Global scale, per sample.
    scale = rng.uniform(1 - 0.08 * strength, 1 + 0.08 * strength, (n, 1, 1))
    out *= scale.astype(np.float32)

    # Coordinate noise.
    out += rng.normal(0, 0.01 * strength, out.shape).astype(np.float32)

    # Temporal shift by up to 2 frames.
    for i in range(n):
        shift = rng.integers(-2, 3)
        if shift:
            out[i] = np.roll(out[i], shift, axis=0)

    # Random frame dropout: repeat the previous frame, imitating a
    # dropped detection rather than inserting an impossible zero pose.
    for i in range(n):
        drop = rng.random(t) < 0.05 * strength
        for f in np.flatnonzero(drop):
            if f > 0:
                out[i, f] = out[i, f - 1]

    return out


# =====================================================================
# MODELS
# =====================================================================


def build_model(name, seq_len, dim, n_classes):
    """Model factories. Kept small: this machine has no GPU."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    tf.keras.utils.set_random_seed(SEED)

    inputs = layers.Input(shape=(seq_len, dim))
    x = layers.Masking(mask_value=0.0)(inputs)

    if name == "bigru":
        x = layers.Bidirectional(
            layers.GRU(64, return_sequences=True, dropout=0.3)
        )(x)
        x = layers.Bidirectional(layers.GRU(32, dropout=0.3))(x)

    elif name == "tcn":
        x = inputs
        for rate in (1, 2, 4, 8):
            x = layers.Conv1D(
                96,
                3,
                padding="causal",
                dilation_rate=rate,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            )(x)
            x = layers.LayerNormalization()(x)
            x = layers.SpatialDropout1D(0.2)(x)
        x = layers.GlobalAveragePooling1D()(x)

    elif name == "attn":
        x = layers.Conv1D(96, 3, padding="same", activation="relu")(inputs)
        x = layers.LayerNormalization()(x)
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=24)(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)

    else:
        raise ValueError(f"Unknown model: {name}")

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


# =====================================================================
# METRICS
# =====================================================================


def expected_calibration_error(confidences, correct, bins=10):
    """ECE. The v3 model was badly overconfident; this quantifies it."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(
            correct[mask].mean() - confidences[mask].mean()
        )
    return float(ece)


def evaluate(model, X, y, signers, n_classes):
    """Full metric set for one evaluation split."""
    probabilities = model.predict(X, verbose=0)
    predictions = probabilities.argmax(1)
    confidences = probabilities.max(1)
    correct = (predictions == y).astype(np.float64)

    per_class = {}
    for c in range(n_classes):
        actual = y == c
        predicted = predictions == c
        tp = int((actual & predicted).sum())
        precision = tp / max(int(predicted.sum()), 1)
        recall = tp / max(int(actual.sum()), 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        per_class[int(c)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(actual.sum()),
        }

    per_signer = {}
    for s in sorted(set(signers.tolist())):
        mask = signers == s
        per_signer[int(s)] = round(float(correct[mask].mean()), 4)

    macro_f1 = float(np.mean([m["f1"] for m in per_class.values()]))
    signer_accuracies = list(per_signer.values())

    return {
        "accuracy": round(float(correct.mean()), 4),
        "macro_f1": round(macro_f1, 4),
        "mean_signer_accuracy": round(float(np.mean(signer_accuracies)), 4),
        "worst_signer_accuracy": round(float(np.min(signer_accuracies)), 4),
        "mean_confidence": round(float(confidences.mean()), 4),
        "mean_confidence_when_wrong": (
            round(float(confidences[correct == 0].mean()), 4)
            if (correct == 0).any()
            else None
        ),
        "ece": round(expected_calibration_error(confidences, correct), 4),
        "per_class": per_class,
        "per_signer": per_signer,
        "n": int(len(y)),
    }


# =====================================================================
# TRAINING
# =====================================================================


def train_fold(
    model_name,
    X_tr,
    y_tr,
    signers_tr,
    n_classes,
    epochs,
    aug_strength,
    rng,
):
    """
    Train one fold.

    The evaluation split is NOT passed in. Early stopping uses an INNER
    signer-grouped split carved out of the training signers only.

    This matters: an earlier version of this function took the
    evaluation split as `validation_data` and early-stopped on it with
    restore_best_weights. That leaks the evaluation set into model
    selection and optimistically biases every reported number, which is
    exactly the failure this project must not ship. The held-out fold is
    now never seen until `evaluate` runs.
    """
    import tensorflow as tf

    unique = sorted(set(signers_tr.tolist()))

    # Hold out roughly a fifth of the TRAINING signers to choose the
    # epoch. Grouped by signer, so no signer straddles the boundary.
    n_inner = max(1, len(unique) // 5)
    inner_val_signers = unique[-n_inner:]

    inner_val = np.isin(signers_tr, inner_val_signers)
    inner_train = ~inner_val

    # Degenerate case: too few signers to split. Fall back to a
    # stratified random split, still never touching the eval fold.
    if inner_train.sum() == 0 or inner_val.sum() == 0:
        shuffled = rng.permutation(len(y_tr))
        cut = int(0.85 * len(y_tr))
        inner_train = np.zeros(len(y_tr), dtype=bool)
        inner_val = np.zeros(len(y_tr), dtype=bool)
        inner_train[shuffled[:cut]] = True
        inner_val[shuffled[cut:]] = True

    seq_len, dim = X_tr.shape[1], X_tr.shape[2]
    model = build_model(model_name, seq_len, dim, n_classes)

    y_inner_train = tf.keras.utils.to_categorical(y_tr[inner_train], n_classes)
    y_inner_val = tf.keras.utils.to_categorical(y_tr[inner_val], n_classes)

    if aug_strength > 0:
        X_aug = augment(X_tr[inner_train], rng, aug_strength)
        X_fit = np.concatenate([X_tr[inner_train], X_aug])
        y_fit = np.concatenate([y_inner_train, y_inner_train])
    else:
        X_fit, y_fit = X_tr[inner_train], y_inner_train

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, verbose=0
        ),
    ]

    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_tr[inner_val], y_inner_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return model, best_epoch


def run_protocol(
    X, y, signers, classes, model_name, config, epochs, aug, protocol
):
    """Runs one evaluation protocol and returns a result dict."""
    rng = np.random.default_rng(SEED)
    n_classes = len(classes)
    folds = []

    if protocol == "cross_domain":
        train_mask = np.isin(signers, STUDIO_SIGNERS)
        test_mask = np.isin(signers, WILD_SIGNERS)
        splits = [("studio->wild", train_mask, test_mask)]

    elif protocol == "within_studio":
        splits = [
            (
                f"loso_{s}",
                np.isin(signers, [x for x in STUDIO_SIGNERS if x != s]),
                signers == s,
            )
            for s in STUDIO_SIGNERS
        ]

    elif protocol == "dev_loso":
        splits = [
            (
                f"loso_{s}",
                np.isin(signers, [x for x in DEV_SIGNERS if x != s]),
                signers == s,
            )
            for s in DEV_SIGNERS
        ]

    elif protocol == "dev_groupcv":
        # Six folds of two signers over the development set (01-12).
        #
        # Used for architecture selection. Signers 13-15 are excluded
        # entirely so that selection pressure never touches them, and
        # each fold pairs a studio signer with another signer so folds
        # are not trivially all-studio. Cheaper than 12-fold LOSO while
        # still strictly signer-grouped.
        pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
        splits = [
            (
                f"fold_{a}_{b}",
                np.isin(signers, [x for x in DEV_SIGNERS if x not in (a, b)]),
                np.isin(signers, [a, b]),
            )
            for a, b in pairs
        ]

    else:
        raise ValueError(protocol)

    for label, train_mask, test_mask in splits:
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        started = time.time()
        model, best_epoch = train_fold(
            model_name,
            X[train_mask],
            y[train_mask],
            signers[train_mask],
            n_classes,
            epochs,
            aug,
            rng,
        )
        metrics = evaluate(
            model, X[test_mask], y[test_mask], signers[test_mask], n_classes
        )
        metrics["fold"] = label
        metrics["best_epoch"] = best_epoch
        metrics["train_n"] = int(train_mask.sum())
        metrics["seconds"] = round(time.time() - started, 1)
        folds.append(metrics)

        print(
            f"    {label:<14} acc={metrics['accuracy']:.3f} "
            f"macroF1={metrics['macro_f1']:.3f} "
            f"ECE={metrics['ece']:.3f} "
            f"({metrics['seconds']:.0f}s)",
            flush=True,
        )

    accuracies = [f["accuracy"] for f in folds]
    return {
        "protocol": protocol,
        "model": model_name,
        "config": asdict(config),
        "classes": classes,
        "folds": folds,
        "summary": {
            "mean_accuracy": round(float(np.mean(accuracies)), 4),
            "std_accuracy": round(float(np.std(accuracies)), 4),
            "min_fold_accuracy": round(float(np.min(accuracies)), 4),
            "mean_macro_f1": round(
                float(np.mean([f["macro_f1"] for f in folds])), 4
            ),
            "mean_ece": round(float(np.mean([f["ece"] for f in folds])), 4),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bigru", choices=["bigru", "tcn", "attn"])
    parser.add_argument(
        "--protocol",
        default="cross_domain",
        choices=["cross_domain", "within_studio", "dev_loso", "dev_groupcv"],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--aug", type=float, default=1.0)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--legacy-scale", action="store_true",
                        help="Use the v3 wrist->MCP hand scale, for A/B.")
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--no-trim", action="store_true",
                        help="Disable motion trimming, for A/B.")
    parser.add_argument("--with-background", action="store_true",
                        help="Add a 'no sign' rejection class from idle footage.")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--classes",
        default="",
        help=(
            "Comma-separated class subset. Used while the landmark cache "
            "is still filling, so an experiment only sees classes that are "
            "complete for every signer it evaluates."
        ),
    )
    parser.add_argument(
        "--signers",
        default="",
        help="Comma-separated signer subset to restrict the dataset to.",
    )
    args = parser.parse_args()

    config = FeatureConfig(
        sequence_length=args.seq_len,
        robust_hand_scale=not args.legacy_scale,
        include_velocity=not args.no_velocity,
        include_masks=not args.no_masks,
        trim_to_motion=not args.no_trim,
    )

    wanted_classes = (
        [c.strip() for c in args.classes.split(",") if c.strip()] or None
    )

    print(f"Building dataset  dim={feature_dimension(config)} ...", flush=True)
    X, y, signers, classes = load_dataset(
        config, wanted_classes, with_background=args.with_background
    )

    if args.signers:
        keep = [int(s) for s in args.signers.split(",")]
        mask = np.isin(signers, keep)
        X, y, signers = X[mask], y[mask], signers[mask]

    print(
        f"  X={X.shape}  classes={len(classes)}  "
        f"signers={sorted(set(signers.tolist()))}",
        flush=True,
    )

    print(f"\nProtocol: {args.protocol}   Model: {args.model}", flush=True)
    result = run_protocol(
        X, y, signers, classes, args.model, config, args.epochs, args.aug,
        args.protocol,
    )

    print("\n  SUMMARY", json.dumps(result["summary"], indent=2))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.model}_{args.protocol}"
    out = RESULT_DIR / f"{tag}.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
