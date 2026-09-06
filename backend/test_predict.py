"""
test_predict.py

Standalone script to sanity-check the /api/v1/ksl/predict endpoint
with a correctly-shaped dummy request (30 frames x 1662 features,
all zeros). This does NOT test recognition accuracy — it only checks
that the endpoint accepts a well-formed request and returns a
structured response. Run the server first (see SETUP notes), then
run this script separately.

Usage (from anywhere, with the server running):
    python test_predict.py
"""
import json
import urllib.request

URL = "http://127.0.0.1:8000/api/v1/ksl/predict"
SEQUENCE_LENGTH = 30
FEATURE_LENGTH = 1662


def main() -> None:
    payload = {"frames": [[0.0] * FEATURE_LENGTH for _ in range(SEQUENCE_LENGTH)]}
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"Status: {response.status}")
            print(json.dumps(json.loads(response.read()), indent=2))
    except urllib.error.HTTPError as exc:
        print(f"HTTP error {exc.code}: {exc.read().decode('utf-8')}")
    except urllib.error.URLError as exc:
        print(f"Could not reach {URL}: {exc.reason}")
        print("Is the server running? uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()