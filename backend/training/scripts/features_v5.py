"""
Fadhili v5 feature construction, built on the raw landmark cache.

This module turns cached raw MediaPipe output into model input tensors.
Every design choice below is a response to a measured failure of the
v2/v3/v4 representation, not a speculative improvement.

What was wrong with the 150-d v3 representation
-----------------------------------------------

1. FRAGILE HAND SCALE.
   normalize_hand() divided all 63 hand values by the *projected*
   wrist -> middle-finger-MCP distance. That distance collapses toward
   zero whenever the hand points at or away from the camera, so the
   normalized coordinates blow up. Seated in-the-wild signers foreshorten
   far more than standing studio signers, which put signers 14/15 at
   mean |x| = 0.75 against a studio training range of 0.37-0.59 —
   outside the training distribution entirely. The model collapsed:
   47 of signer 14's 50 predictions became "Market" at 0.96 confidence.

   v5 uses the mean distance of all 21 landmarks from the hand centroid.
   Averaging over 21 points is far more stable under foreshortening and
   partial occlusion than a single landmark pair.

2. ANISOTROPIC COORDINATES.
   MediaPipe normalizes x by frame width and y by frame height. On
   1920x1080 that stretches every shape horizontally by 16/9. v5
   rescales x into height units so geometry is isotropic.

3. ZEROS WERE AMBIGUOUS.
   A missing hand and a hand resting exactly at the origin produced the
   same values. With hand-detection failure ranging from 7% to 62%
   across signers, that ambiguity is severe. v5 adds explicit
   presence masks and interpolates short gaps.

4. HAND LOCATION WAS DISCARDED.
   Translating each hand to its own wrist throws away *where* the sign
   happens, which is phonemically contrastive in KSL. v5 keeps wrist
   position in shoulder-normalized body space as separate features.

Feature layout (per frame), with default config
-----------------------------------------------
    pose xyz            6 landmarks x 3   = 18
    pose visibility     6                 =  6
    left hand shape     21 x 3            = 63
    right hand shape    21 x 3            = 63
    left wrist in body space               3
    right wrist in body space              3
    left hand scale ratio                  1
    right hand scale ratio                 1
    left present mask                      1
    right present mask                     1
                                          ---
                                          160

Velocity, when enabled, appends first differences of the continuous
channels (masks excluded), doubling nothing that is categorical.

Nothing here re-runs MediaPipe. Building a full dataset from the raw
cache takes seconds, so representations can be ablated properly.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO / "backend" / "training" / "data" / "raw_landmarks"

# Upper-body pose landmarks: shoulders, elbows, wrists.
POSE_IDS = [11, 12, 13, 14, 15, 16]

# Indices within POSE_IDS.
L_SHOULDER, R_SHOULDER = 0, 1
L_WRIST_POSE, R_WRIST_POSE = 4, 5

EPS = 1e-6


@dataclass
class FeatureConfig:
    """Every knob that defines a representation, for reproducibility."""

    sequence_length: int = 32
    aspect_correct: bool = True
    robust_hand_scale: bool = True
    interpolate_gaps: bool = True
    max_gap: int = 5
    include_masks: bool = True
    include_wrist_position: bool = True
    include_scale_ratio: bool = True
    include_velocity: bool = True
    clip_value: float = 5.0

    # Crop to the actively-signing interval before resampling.
    #
    # Measured motion profiles show a large temporal domain shift: studio
    # signers 01-10 are actively moving in 34-44% of frames, while wild
    # signers idle far more (signer 13 32.7%, signer 15 27.3%, signer 14
    # just 26.0% at less than half the studio wrist velocity). Uniform
    # resampling of a mostly-idle recording spends most of its frames on
    # a stationary signer and compresses the sign itself into a handful,
    # so the same sign occupies a different portion of the tensor
    # depending on who performed it. Trimming removes that confound.
    trim_to_motion: bool = True
    motion_energy_keep: float = 0.90
    motion_pad_frames: int = 4

    def describe(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _aspect_scale(meta, enabled):
    """
    Factor that converts x from width-normalized to height-normalized
    units, so x and y become directly comparable.
    """
    if not enabled:
        return 1.0
    width, height = float(meta[1]), float(meta[2])
    if height <= 0 or width <= 0:
        return 1.0
    return width / height


def _interpolate_missing(seq, present, max_gap):
    """
    Linearly interpolate landmark frames across short detection gaps.

    Only gaps of at most `max_gap` frames bounded by detections on both
    sides are filled. Long dropouts and dropouts at the sequence edges
    are left as missing, because inventing a full second of hand motion
    would be fabricating data the model then trains on.
    """
    seq = seq.copy()
    present = present.copy()
    n = len(present)

    if present.sum() == 0:
        return seq, present

    idx = np.flatnonzero(present)

    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if gap <= 0 or gap > max_gap:
            continue
        for step in range(1, gap + 1):
            t = step / (gap + 1)
            seq[a + step] = (1 - t) * seq[a] + t * seq[b]
            present[a + step] = True

    return seq, present


def _normalize_hand(hand, scale_factor, robust):
    """
    Translate a hand to its wrist and scale it to unit size.

    Returns (63-vector, scale) where `scale` is the hand's size in
    aspect-corrected image units before normalization.
    """
    pts = hand.astype(np.float32).copy()
    pts[:, 0] *= scale_factor

    wrist = pts[0].copy()
    centered = pts - wrist

    if robust:
        # Mean distance of all landmarks from the hand centroid, in 2D.
        # Averaging 21 points is stable under foreshortening; the single
        # wrist->MCP pair that v3 used is not.
        centroid = pts[:, :2].mean(axis=0)
        scale = np.linalg.norm(pts[:, :2] - centroid, axis=1).mean()
    else:
        scale = np.linalg.norm(pts[9, :2] - pts[0, :2])

    if scale > EPS:
        centered = centered / scale

    return centered.reshape(-1), float(scale)


def _motion_interval(pose, pose_ok, keep, pad):
    """
    Find the frame interval containing the central `keep` fraction of
    cumulative wrist motion energy.

    Wrist positions are expressed in shoulder widths so the measure does
    not depend on how far the signer sits from the camera. Returns
    (start, end) frame indices, end exclusive.
    """
    n = pose.shape[0]
    if n < 4:
        return 0, n

    left_shoulder = pose[:, 11, :2]
    right_shoulder = pose[:, 12, :2]
    centre = (left_shoulder + right_shoulder) / 2.0

    shoulder = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    shoulder[shoulder < EPS] = np.nan

    left_wrist = (pose[:, 15, :2] - centre) / shoulder[:, None]
    right_wrist = (pose[:, 16, :2] - centre) / shoulder[:, None]

    velocity = np.linalg.norm(
        np.diff(left_wrist, axis=0), axis=1
    ) + np.linalg.norm(np.diff(right_wrist, axis=0), axis=1)

    velocity = np.nan_to_num(velocity)
    velocity[~pose_ok[1:]] = 0.0

    total = velocity.sum()
    if total <= EPS:
        return 0, n

    cumulative = np.cumsum(velocity) / total
    margin = (1.0 - keep) / 2.0

    start = int(np.searchsorted(cumulative, margin))
    end = int(np.searchsorted(cumulative, 1.0 - margin)) + 1

    start = max(0, start - pad)
    end = min(n, end + pad)

    # Never trim away so much that nothing meaningful survives.
    if end - start < 8:
        return 0, n

    return start, end


def _resample(array, length):
    """Resample a (T, D) sequence to exactly `length` frames."""
    t = array.shape[0]
    if t == length:
        return array
    if t == 0:
        return np.zeros((length, array.shape[1]), dtype=np.float32)
    src = np.linspace(0.0, t - 1.0, length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo).astype(np.float32)[:, None]
    return (1 - w) * array[lo] + w * array[hi]


def build_sequence(npz_path, config):
    """
    Build one model-ready sequence of shape (sequence_length, D).

    Returns None if the file holds no usable pose.
    """
    with np.load(npz_path) as data:
        pose = data["pose"].astype(np.float32)
        left = data["left"].astype(np.float32)
        right = data["right"].astype(np.float32)
        pose_ok = data["pose_ok"]
        left_ok = data["left_ok"].copy()
        right_ok = data["right_ok"].copy()
        meta = data["meta"]

    if pose.shape[0] == 0 or pose_ok.sum() == 0:
        return None

    if config.trim_to_motion:
        start, end = _motion_interval(
            pose,
            pose_ok,
            config.motion_energy_keep,
            config.motion_pad_frames,
        )
        pose = pose[start:end]
        left = left[start:end]
        right = right[start:end]
        pose_ok = pose_ok[start:end]
        left_ok = left_ok[start:end]
        right_ok = right_ok[start:end]

    ax = _aspect_scale(meta, config.aspect_correct)

    if config.interpolate_gaps:
        left, left_ok = _interpolate_missing(left, left_ok, config.max_gap)
        right, right_ok = _interpolate_missing(right, right_ok, config.max_gap)

    frames = []

    for t in range(pose.shape[0]):
        row = []

        # ---- pose, shoulder-centred and shoulder-scaled ----
        p = pose[t][POSE_IDS]
        coords = p[:, :3].copy()
        coords[:, 0] *= ax
        visibility = p[:, 3]

        centre = (coords[L_SHOULDER] + coords[R_SHOULDER]) / 2.0
        shoulder = np.linalg.norm(
            coords[L_SHOULDER, :2] - coords[R_SHOULDER, :2]
        )

        if pose_ok[t] and shoulder > EPS:
            body = (coords - centre) / shoulder
        else:
            body = np.zeros_like(coords)
            shoulder = 0.0

        row.append(body.reshape(-1))
        row.append(visibility)

        # ---- hands ----
        for hand, ok in ((left[t], left_ok[t]), (right[t], right_ok[t])):
            if ok:
                shape, hscale = _normalize_hand(
                    hand, ax, config.robust_hand_scale
                )
            else:
                shape, hscale = np.zeros(63, dtype=np.float32), 0.0
            row.append(shape)

            if config.include_scale_ratio:
                # Hand size relative to shoulder width encodes how far
                # the signer is from the camera without depending on
                # the brittle absolute scale.
                ratio = hscale / shoulder if shoulder > EPS else 0.0
                row.append(np.array([ratio], dtype=np.float32))

        # ---- wrist position in body space ----
        if config.include_wrist_position:
            row.append(body[L_WRIST_POSE])
            row.append(body[R_WRIST_POSE])

        if config.include_masks:
            row.append(
                np.array(
                    [float(left_ok[t]), float(right_ok[t])],
                    dtype=np.float32,
                )
            )

        frames.append(np.concatenate(row).astype(np.float32))

    seq = np.stack(frames)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    seq = np.clip(seq, -config.clip_value, config.clip_value)

    seq = _resample(seq, config.sequence_length)

    if config.include_velocity:
        velocity = np.diff(seq, axis=0, prepend=seq[:1])
        seq = np.concatenate([seq, velocity], axis=1)

    return seq.astype(np.float32)


def feature_dimension(config):
    """Dimension implied by a config, without touching the cache."""
    d = 18 + 6 + 63 + 63
    if config.include_scale_ratio:
        d += 2
    if config.include_wrist_position:
        d += 6
    if config.include_masks:
        d += 2
    if config.include_velocity:
        d *= 2
    return d
