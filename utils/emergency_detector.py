"""
emergency_detector.py
-----------------------
Highest-priority safety check (requirement 4-v):
"If anything is like terrorism/anti national calls it should be given as emergency"

This module runs BEFORE and INDEPENDENT of the purpose classifier.
If it fires, the final "Purpose" label is forced to "Emergency" regardless
of what the ML classifier predicts, and a prominent warning is shown in the UI.

This is a deliberately simple, high-recall keyword/phrase system:
false positives (flagging something borderline) are far cheaper than
false negatives in this domain, so it is tuned to be sensitive.
"""

import re

EMERGENCY_KEYWORDS = [
    "bomb", "explosive", "explosion", "terrorist", "terrorism", "hijack",
    "attack the", "attack on", "kill the president", "assassinate",
    "mass shooting", "shoot up", "anti national", "anti-national",
    "overthrow the government", "chemical weapon", "biological weapon",
    "suicide vest", "detonate", "planted a bomb", "gun down", "jihad attack",
    "kidnap", "hostage", "blow up", "national assembly attack",
    "railway station attack", "airport attack", "coordinated attack",
    "extremist cell", "sleeper cell", "radicalize", "insurgent attack",
]

# Regex for near-miss phrasing (e.g., "planning an attack", "plant a bomb")
EMERGENCY_PATTERNS = [
    re.compile(r"\bplan(?:ning|ned)?\s+(?:an?\s+)?attack\b", re.IGNORECASE),
    re.compile(r"\bplant(?:ing|ed)?\s+(?:a\s+)?bomb\b", re.IGNORECASE),
    re.compile(r"\bdestroy\s+the\s+(?:power grid|government|capital|building)\b", re.IGNORECASE),
    re.compile(r"\b(weapons?|explosives?)\s+(?:delivered|delivery|for the operation)\b", re.IGNORECASE),
]


def check_emergency(text: str) -> dict:
    """
    Returns {
        'is_emergency': bool,
        'matched_terms': [list of matched keywords/phrases],
    }
    """
    if not text:
        return {"is_emergency": False, "matched_terms": []}

    lowered = text.lower()
    matched = [kw for kw in EMERGENCY_KEYWORDS if kw in lowered]

    for pattern in EMERGENCY_PATTERNS:
        if pattern.search(text):
            matched.append(pattern.pattern)

    matched = sorted(set(matched))
    return {"is_emergency": len(matched) > 0, "matched_terms": matched}
