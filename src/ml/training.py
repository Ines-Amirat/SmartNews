from typing import Tuple

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.infra.io import load_dataset
from src.nlp.preprocess import clean_text
from src.ml.pipeline import create_pipeline
from src.core.config import MODEL_PATH


def prepare_data(test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Load and preprocess the dataset, then split into train/test.
    Returns: X_train, X_test, y_train, y_test
    """
    df = load_dataset()

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model() -> Tuple:
    """
    Train the SmartNews model and return:
    - trained model
    - evaluation report on test data (as string)
    """
    X_train, X_test, y_train, y_test = prepare_data()

    model = create_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    return model, report


def train_and_save_model() -> None:
    """
    Train the model, print a classification report,
    and save the trained pipeline to disk.
    """
    print("🔄 Training SmartNews model...")
    model, report = train_model()

    print("✅ Training finished. Evaluation report:")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")
