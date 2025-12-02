from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def create_pipeline() -> Pipeline:
    """
    Create an ML pipeline for text classification:
    - TF-IDF vectorizer
    - Linear SVM classifier
    """
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=20000
            )),
            ("clf", LinearSVC())
        ]
    )
    return pipeline
