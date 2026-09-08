from pathlib import Path
from keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "training"
    / "models"
    / "fadhili_ksl_v2.keras"
)

print("=" * 70)
print("FADHILI V2 MODEL CHECK")
print("=" * 70)
print(f"Model: {MODEL_PATH}")

model = load_model(MODEL_PATH)

print()
model.summary()

print()
print("=" * 70)
print(f"TOTAL PARAMETERS: {model.count_params():,}")
print("=" * 70)