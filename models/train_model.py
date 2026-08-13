"""
train_model.py
----------------
Trains the "Purpose" classifier (requirement 3):
    Family Call, Friend Call, Complaint Call, Business Call, Fraud Call
    (Emergency is handled separately by utils/emergency_detector.py's
     rule-based override, since it must NEVER be missed / ML-only isn't
     reliable enough for that safety-critical case.)

Pipeline: TF-IDF (word + char n-grams) -> Logistic Regression
Chosen for: fast to train, interpretable coefficients, works well on
small/medium labeled text datasets like the one in data/calls_dataset.csv.

Usage:
    python train_model.py
Produces:
    models/purpose_classifier.joblib   (sklearn Pipeline, ready for inference)
    models/label_report.txt            (classification report on held-out split)
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "calls_dataset.csv")
MODEL_PATH = os.path.join(HERE, "purpose_classifier.joblib")
REPORT_PATH = os.path.join(HERE, "label_report.txt")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run data/prepare_dataset.py first."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "purpose"])
    # Emergency rows are excluded from the ML classifier's training set on purpose:
    # emergency detection is rule-based (see utils/emergency_detector.py) for
    # maximum recall on a safety-critical category. Including only a handful
    # of examples in an ML classifier would give false confidence.
    df = df[df["purpose"] != "Emergency"]
    return df


def train():
    df = load_data()
    X = df["text"]
    y = df["purpose"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
