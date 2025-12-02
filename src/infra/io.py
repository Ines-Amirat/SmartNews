import pandas as pd
from src.core.config import DATA_PATH


def load_dataset() -> pd.DataFrame:
    """
    Load the news dataset from CSV file.
    Expects a CSV with at least two columns: 'text' and 'label'.
    """
    df = pd.read_csv(DATA_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    return df
