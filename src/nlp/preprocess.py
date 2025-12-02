import re


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - lowercase
    - remove non-alphanumeric characters
    - normalize spaces
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    # Keep letters, digits, and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text
