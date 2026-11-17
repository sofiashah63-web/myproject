"""Central place for paths and constants used across the project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset.csv"

MODEL_PATH = BASE_DIR / "disease_model.pkl"
RANDOM_STATE = 42
SAMPLES_PER_DISEASE = 40   # synthetic "partial symptom" samples per disease
MIN_SYMPTOM_FRACTION = 0.4  # a sample keeps at least 40% of a disease's symptoms
TOP_K = 3                   # number of predictions shown in the app
