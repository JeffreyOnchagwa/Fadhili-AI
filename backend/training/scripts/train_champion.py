"""
Train and freeze the Fadhili KSL champion model.

Protocol
--------
Trained on development signers 01-12 only. Signers 13-15 are evaluated
exactly ONCE, after training, and are never used for early stopping,
model selection, threshold tuning or calibration. They remain a
held-out DIAGNOSTIC set rather than a pristine test set, because they
were inspected during v2/v3/v4 development — that history cannot be
undone, and calling them untouched would overstate what we know.

Early stopping uses a signer-grouped split WITHIN the development set
(signers 11-12 held out), so no signer is ever both trained on and used
to choose an epoch.

Rejection
---------
A "no sign" class is trained from the idle stretches of real recordings
— the footage before and after each sign, which motion trimming
discards. This is genuine "nobody is signing" data from the same
cameras and rooms, so the model can answer "no confident sign detected"
instead of forcing noise into a vocabulary word.

Calibration
-----------
Temperature scaling is fitted on the held-out development signers, not
on 13-15. The v3 model reported ~0.93 confidence on predictions that
were wrong; a single temperature meaningfully reduces that.

Outputs
-------
    models/fadhili_ksl_v5_champion.keras
    metadata/fadhili_v5_champion.json   (classes, config, metrics, T)
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from experiments_v5 import (  # noqa: E402
    DEV_SIGNERS,
    DIAGNOSTIC_SIGNERS,
    NO_SIGN_LABEL,
    SEED,
    augment,
    build_model,
    evaluate,
    load_dataset,
)
from features_v5 import FeatureConfig, feature_dimension  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO / "backend" / "training" / "models"
META_DIR = REPO / "backend" / "training" / "data" / "metadata"


def fit_temperature(logits, labels):
    """
    Fit a single temperature by minimising NLL on held-out development
    data. Standard temperature scaling: it cannot change which class
    wins, only how confident the model claims to be.
    """
    import tensorflow as tf

    logits_t = tf.constant(logits, dtype=tf.float32)
    labels_t = tf.constant(labels, dtype=tf.int32)

    log_temperature = tf.Variable(0.0, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(0.02)

    for _ in range(300):
        with tf.GradientTape() as tape:
            temperature = tf.exp(log_temperature)
            scaled = logits_t / temperature
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels_t, logits=scaled
                )
            )
        grads = tape.gradient(loss, [log_temperature])
        optimizer.apply_gradients(zip(grads, [log_temperature]))

    return float(np.exp(log_temperature.numpy()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bigru")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--classes", default="")
    parser.add_argument("--no-background", action="store_true")
    args = parser.parse_args()

    import tensorflow as tf

    config = FeatureConfig(sequence_length=args.seq_len)
    wanted = [c.strip() for c in args.classes.split(",") if c.strip()] or None

    print(f"Building dataset (dim={feature_dimension(config)}) ...", flush=True)
    X, y, signers, classes = load_dataset(
        config, wanted, with_background=not args.no_background
    )
    n_classes = len(classes)
    print(f"  X={X.shape}  classes={n_classes}", flush=True)
    for index, name in enumerate(classes):
        print(f"    {index:>2} {name:<14} n={(y == index).sum()}")

    # Signer-grouped splits. 11-12 choose the epoch; 13-15 are untouched.
    fit_signers = [s for s in DEV_SIGNERS if s not in (11, 12)]
    train_mask = np.isin(signers, fit_signers)
    stop_mask = np.isin(signers, [11, 12])
    diag_mask = np.isin(signers, DIAGNOSTIC_SIGNERS)

    print(
        f"\n  train signers {fit_signers} -> {train_mask.sum()} samples"
        f"\n  early-stop signers [11, 12] -> {stop_mask.sum()} samples"
        f"\n  diagnostic signers {DIAGNOSTIC_SIGNERS} -> {diag_mask.sum()}"
        " samples (evaluated once)",
        flush=True,
    )

    rng = np.random.default_rng(SEED)
    model = build_model(args.model, X.shape[1], X.shape[2], n_classes)
    print(f"\n  parameters: {model.count_params():,}", flush=True)

    y_train = tf.keras.utils.to_categorical(y[train_mask], n_classes)
    y_stop = tf.keras.utils.to_categorical(y[stop_mask], n_classes)

    X_aug = augment(X[train_mask], rng, 1.0)
    X_fit = np.concatenate([X[train_mask], X_aug])
    y_fit = np.concatenate([y_train, y_train])

    started = time.time()
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X[stop_mask], y_stop),
        epochs=args.epochs,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8, verbose=0
            ),
        ],
        verbose=2,
    )
    minutes = (time.time() - started) / 60
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    # ---- calibration, fitted on held-out DEV signers only ----
    logit_model = tf.keras.Model(
        model.input, model.layers[-1].input
    )  # pre-softmax
    dense = model.layers[-1]
    stop_logits = (
        logit_model.predict(X[stop_mask], verbose=0) @ dense.kernel.numpy()
        + dense.bias.numpy()
    )
    temperature = fit_temperature(stop_logits, y[stop_mask])

    print(f"\n  trained {minutes:.1f} min, best epoch {best_epoch}")
    print(f"  fitted temperature: {temperature:.3f}")

    # ---- evaluation ----
    results = {
        "model_name": f"Fadhili_KSL_v5_{args.model}",
        "architecture": args.model,
        "parameters": int(model.count_params()),
        "classes": classes,
        "feature_config": asdict(config),
        "feature_dimension": int(X.shape[2]),
        "sequence_length": int(X.shape[1]),
        "temperature": temperature,
        "train_signers": fit_signers,
        "early_stop_signers": [11, 12],
        "diagnostic_signers": DIAGNOSTIC_SIGNERS,
        "training_minutes": round(minutes, 1),
        "best_epoch": best_epoch,
        "seed": SEED,
        "evaluation_note": (
            "Signers 13-15 are a held-out DIAGNOSTIC set, not a pristine "
            "test set: they were inspected during v2/v3/v4 development. "
            "They were not used for training, early stopping, calibration "
            "or model selection here."
        ),
    }

    print("\n  early-stop signers 11-12:")
    results["dev_holdout"] = evaluate(
        model, X[stop_mask], y[stop_mask], signers[stop_mask], n_classes
    )
    print(f"    accuracy {results['dev_holdout']['accuracy']}")

    print("  diagnostic signers 13-15 (single evaluation):")
    results["diagnostic"] = evaluate(
        model, X[diag_mask], y[diag_mask], signers[diag_mask], n_classes
    )
    diag = results["diagnostic"]
    print(f"    accuracy      {diag['accuracy']}")
    print(f"    macro F1      {diag['macro_f1']}")
    print(f"    worst signer  {diag['worst_signer_accuracy']}")
    print(f"    ECE           {diag['ece']}")

    # Rejection quality, if a no-sign class exists.
    if NO_SIGN_LABEL in classes:
        idx = classes.index(NO_SIGN_LABEL)
        probs = model.predict(X[diag_mask], verbose=0)
        pred = probs.argmax(1)
        actual = y[diag_mask]
        is_bg = actual == idx
        results["rejection"] = {
            "no_sign_recall": round(float((pred[is_bg] == idx).mean()), 4),
            "false_reject_rate": round(
                float((pred[~is_bg] == idx).mean()), 4
            ),
            "n_no_sign": int(is_bg.sum()),
        }
        print(
            f"    no-sign recall {results['rejection']['no_sign_recall']}, "
            f"false-reject {results['rejection']['false_reject_rate']}"
        )

    # ---- save ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "fadhili_ksl_v5_champion.keras"
    model.save(model_path)

    META_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = META_DIR / "fadhili_v5_champion.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\n  saved {model_path.name} ({size_mb:.2f} MB)")
    print(f"  saved {meta_path.name}")
    print("\n  V3 fallback left untouched at fadhili_ksl_v3_candidate.keras")


if __name__ == "__main__":
    main()
