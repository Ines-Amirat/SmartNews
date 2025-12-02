import joblib

from src.nlp.preprocess import clean_text
from src.core.config import MODEL_PATH


def load_model():
    """
    Load the trained SmartNews model from disk.
    """
    model = joblib.load(MODEL_PATH)
    return model


def predict(text: str) -> str:
    """
    Predict the category of a news article text.
    Returns the predicted label (e.g., 'sports', 'world', etc.).
    """
    model = load_model()
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    return prediction
