"""
purpose_classifier.py
-----------------------
Inference-time wrapper around the trained ML model (models/purpose_classifier.joblib).
Combines it with the rule-based emergency override so that:

    Emergency detector fires  -> purpose is FORCED to "Emergency" (safety first)
    Otherwise                 -> ML model's predicted class is used

This is the single entry point the Streamlit app should call.
"""

import os
import joblib

from utils.emergency_detector import check_emergency

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "models", "purpose_classifier.joblib")

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                f"Run: python data/prepare_dataset.py && python models/train_model.py"
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def classify_purpose(text: str) -> dict:
    """
    Returns {
        'purpose': str,
        'confidence': float,
        'is_emergency': bool,
        'emergency_terms': [...],
        'class_probabilities': {class: prob, ...}
    }
    """
    emergency_result = check_emergency(text)

    model = _load_model()
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    class_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    best_idx = proba.argmax()
    ml_purpose = classes[best_idx]
    ml_confidence = float(proba[best_idx])

    if emergency_result["is_emergency"]:
        final_purpose = "Emergency"
        final_confidence = 1.0
    else:
        final_purpose = ml_purpose
        final_confidence = ml_confidence

    return {
        "purpose": final_purpose,
        "confidence": round(final_confidence, 3),
        "is_emergency": emergency_result["is_emergency"],
        "emergency_terms": emergency_result["matched_terms"],
        "class_probabilities": class_probs,
    }
