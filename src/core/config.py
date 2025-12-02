from pathlib import Path

# Root of the project (assumes this file is under src/core/)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "raw" / "news.csv"
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
