"""
speaker_detector.py
--------------------
Estimates the number of people in the call transcript:
    1  -> single person (monologue / voicemail / one-sided note)
    2  -> two-person conversation
    3+ -> conference call (multiple participants)

Detection strategy (heuristic, explainable, no external API needed):
 1. Look for explicit speaker labels like "Speaker 1:", "John:", "Agent:", "Caller:"
    at the start of lines - the most reliable signal when present.
 2. If no labels are found, fall back to a lightweight turn-taking /
    pronoun-shift heuristic on sentence structure to guess 1 vs 2+.
"""

import re
from collections import OrderedDict

SPEAKER_LABEL_PATTERN = re.compile(
    r"^\s*(?:\[?\s*)([A-Za-z][A-Za-z0-9 _.\-]{0,25}?)\s*(?:\]?\s*)[:\-]\s+", re.MULTILINE
)

GENERIC_LABELS = {
    "speaker", "speaker1", "speaker2", "speaker 1", "speaker 2", "agent",
    "caller", "customer", "rep", "representative", "user", "client",
}


def detect_speakers(text: str) -> dict:
    """
    Returns {
        'num_speakers': int,
        'speaker_labels': [list of distinct labels found],
        'method': 'explicit_labels' | 'heuristic',
        'call_type': 'Single Person' | 'Two Person' | 'Conference Call'
    }
    """
    if not text or not text.strip():
        return {
            "num_speakers": 0,
            "speaker_labels": [],
            "method": "none",
            "call_type": "Unknown",
        }

    matches = SPEAKER_LABEL_PATTERN.findall(text)
    labels = OrderedDict()
    for m in matches:
        key = m.strip().lower()
        if 0 < len(key) <= 30:
            labels[key] = m.strip()

    if len(labels) >= 1:
        num = len(labels)
        method = "explicit_labels"
    else:
        # Heuristic fallback: count paragraph/line breaks as weak turn-taking signal,
        # and look for dialogue-ish patterns (question then response cues).
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        turn_like_lines = sum(1 for l in lines if len(l) < 200)
        question_marks = text.count("?")

        if len(lines) <= 1 and question_marks == 0:
            num = 1
        elif question_marks >= 1 and turn_like_lines >= 2:
            num = 2
        else:
            num = 1
        method = "heuristic"

    if num <= 1:
        call_type = "Single Person"
    elif num == 2:
        call_type = "Two Person"
    else:
        call_type = "Conference Call"

    return {
        "num_speakers": max(num, 1),
        "speaker_labels": list(labels.values()),
        "method": method,
        "call_type": call_type,
    }
