"""Konfigurasi global proyek."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "indo_spam.csv"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"

RANDOM_SEED = 42
TEST_SIZE = 0.2

LABELS = ["Normal", "Spam"]      # urutan alfabetis = urutan classes_ di scikit-learn
POS_LABEL = "Spam"

MODEL_NAMES = {
    "nb": "Naive Bayes",
    "lr": "Logistic Regression",
}
EXPLAIN_MODEL = "lr"             # model linear yang dipakai untuk penjelasan kata
