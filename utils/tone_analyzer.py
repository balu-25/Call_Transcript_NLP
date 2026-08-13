"""
tone_analyzer.py
-----------------
Classifies the emotional/vocal "tone" of a transcript into:
    High    -> agitated, urgent, angry, excited, loud (lots of caps/exclaim/intensity words)
    Low     -> subdued, sad, quiet, hesitant
    Neutral -> calm, matter-of-fact, ordinary conversation

Since we only have TEXT (no audio), tone here is a linguistic proxy based on:
    - punctuation intensity (!, ALL CAPS, repeated punctuation)
    - sentiment polarity + subjectivity (VADER, bundled with nltk)
    - intensity keyword lexicons (urgency / anger vs calm / sadness)

This is intentionally explainable and tunable (no black box) so it is
easy to justify to non-technical stakeholders.
"""

import re
import nltk

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

_SIA = None


def _get_sia():
    global _SIA
    if _SIA is None:
        try:
            _SIA = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _SIA = SentimentIntensityAnalyzer()
    return _SIA


HIGH_INTENSITY_WORDS = {
    "immediately", "urgent", "now", "unacceptable", "demand", "furious",
    "angry", "asap", "emergency", "help", "stop", "hurry", "danger",
    "critical", "shouting", "yell", "outrageous", "disgusted", "terrible",
    "worst", "horrible", "scream", "attack", "explosion", "threat",
}

LOW_INTENSITY_WORDS = {
    "tired", "sad", "sorry", "quiet", "unsure", "maybe", "hesitant",
    "worried", "depressed", "lonely", "miss", "sigh", "unwell", "sick",
    "exhausted", "down", "low", "regret", "grief", "loss",
}


def _punctuation_intensity(text: str) -> float:
    exclaims = text.count("!")
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", text))
    repeated_punct = len(re.findall(r"[!?]{2,}", text))
    words = max(len(text.split()), 1)
    score = (exclaims * 1.0 + caps_words * 1.5 + repeated_punct * 2.0) / words
    return score


def _keyword_score(text: str, vocab: set) -> int:
    lowered = re.findall(r"[a-z']+", text.lower())
    return sum(1 for w in lowered if w in vocab)


def classify_tone(text: str) -> dict:
    """
    Returns {
        'tone': 'High' | 'Neutral' | 'Low',
        'confidence': float,
        'details': {...}
    }
    """
    if not text or not text.strip():
        return {"tone": "Neutral", "confidence": 0.0, "details": {}}

    sia = _get_sia()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]  # -1 (neg) to +1 (pos)

    punct_intensity = _punctuation_intensity(text)
    high_hits = _keyword_score(text, HIGH_INTENSITY_WORDS)
    low_hits = _keyword_score(text, LOW_INTENSITY_WORDS)

    # Combine signals into a single "arousal/intensity" score
    intensity = punct_intensity * 5 + high_hits * 0.8 - low_hits * 0.3
    negativity = -compound  # more negative sentiment -> higher when combined with low energy

    details = {
        "vader_compound": round(compound, 3),
        "punctuation_intensity": round(punct_intensity, 3),
        "high_intensity_hits": high_hits,
        "low_intensity_hits": low_hits,
    }

    if intensity > 0.6 or high_hits >= 2 or scores["neg"] > 0.35:
        tone = "High"
        confidence = min(0.95, 0.5 + intensity / 3)
    elif low_hits >= 2 or (compound < -0.2 and punct_intensity < 0.05):
        tone = "Low"
        confidence = min(0.9, 0.5 + low_hits * 0.1 + abs(negativity) * 0.2)
    else:
        tone = "Neutral"
        confidence = 0.6 + (0.2 if abs(compound) < 0.2 else 0)

    return {"tone": tone, "confidence": round(float(confidence), 2), "details": details}
