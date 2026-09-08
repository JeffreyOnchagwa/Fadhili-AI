"""
Fadhili AI backend smoke test.

Checks the live API end to end against the current architecture:
health, vocabulary, and a real prediction built from cached landmark
data. This replaces test_predict.py / webcam_test.py, which tested a
1662-feature wire format from before the v3 rewrite and no longer
matched anything the API accepts.

This is a smoke test, not an accuracy benchmark — it confirms the
server is up, the champion model loaded, and a well-formed request
gets a well-formed response. Model quality is measured separately by
backend/training/scripts/experiments_v5.py.

Usage (server must already be running, e.g. `uvicorn app.main:app`):
    python test_api.py [base_url]
"""

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
POSE_IDS = [11, 12, 13, 14, 15, 16]
RAW_ROOT = Path(__file__).resolve().parent / "training" / "data" / "raw_landmarks"


def request_json(path, payload=None, method=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method or ("POST" if data else "GET"),
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.status, json.load(response)


def npz_to_wire_frames(npz_path):
    """Rebuild the raw 150-value wire format from a cached landmark file."""
    data = np.load(npz_path)
    n = data["pose"].shape[0]

    frames = np.zeros((n, 150), dtype=np.float32)
    frames[:, :24] = data["pose"][:, POSE_IDS, :].reshape(n, 24)
    frames[:, 24:87] = np.where(
        data["left_ok"][:, None, None], data["left"], 0
    ).reshape(n, 63)
    frames[:, 87:150] = np.where(
        data["right_ok"][:, None, None], data["right"], 0
    ).reshape(n, 63)

    return frames, float(data["meta"][1]), float(data["meta"][2])


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main():
    print(f"Fadhili AI backend smoke test — {BASE_URL}\n")
    ok = True

    print("health")
    status, health = request_json("/health")
    ok &= check("HTTP 200", status == 200)
    ok &= check("status ok", health.get("status") == "ok")
    ok &= check("model_loaded", health.get("model_loaded") is True)
    ok &= check(
        "model_version present", health.get("model_version") in ("v3", "v5")
    )

    print("\nvocabulary")
    status, classes = request_json("/api/v1/ksl/classes")
    ok &= check("HTTP 200", status == 200)
    class_list = classes.get("classes", [])
    ok &= check(f"non-empty ({len(class_list)} classes)", len(class_list) > 0)

    print("\nv2 predict — real sign")
    sample = next(RAW_ROOT.rglob("*.npz"), None)
    if sample is None:
        print("  [SKIP] no cached landmarks found — run cache_raw_landmarks.py first")
    else:
        frames, width, height = npz_to_wire_frames(sample)
        idx = np.linspace(0, len(frames) - 1, 48).astype(int)
        payload = {
            "frames": np.round(frames[idx], 4).tolist(),
            "width": width,
            "height": height,
        }
        status, result = request_json("/api/v2/ksl/predict", payload)
        ok &= check("HTTP 200", status == 200)
        ok &= check("has 'reason' field", "reason" in result)
        ok &= check("has 'motion_energy' field", "motion_energy" in result)
        print(f"    file={sample.name} truth={sample.parent.name} "
              f"prediction={result.get('prediction')} "
              f"confidence={result.get('confidence')}")

    print("\nv2 predict — malformed input rejected")
    try:
        status, _ = request_json(
            "/api/v2/ksl/predict",
            {"frames": [[0.0] * 149], "width": 1920, "height": 1080},
        )
        ok &= check("HTTP 422 for wrong feature length", status == 422)
    except urllib.error.HTTPError as exc:
        ok &= check("HTTP 422 for wrong feature length", exc.code == 422)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
