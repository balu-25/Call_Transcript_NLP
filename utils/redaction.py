"""
redaction.py
------------
Detects and masks sensitive / confidential information in transcripts
BEFORE it is stored, displayed in logs, or sent to any model.

This directly satisfies requirement 4-iv: "Try not to include
passwords/confidential things (so my app is secure)".

Design notes:
- All redaction happens in-memory, on the client-provided text only.
- Nothing is written to disk unless the user explicitly exports a report,
  and even then the REDACTED version is what gets saved.
- Patterns are intentionally broad (better to over-redact than leak).
"""

import re

# --- Regex patterns for common sensitive data -------------------------------

PATTERNS = {
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "OTP_CODE": re.compile(r"\b\d{4,8}\b(?=\s*(?:otp|code|pin)?)", re.IGNORECASE),
    "CVV": re.compile(r"\bcvv\s*[:\-]?\s*\d{3,4}\b", re.IGNORECASE),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b"),
    "PASSWORD_PHRASE": re.compile(
        r"(password|pwd|passcode|pin number|pin code)\s*(is|:)?\s*\S+", re.IGNORECASE
    ),
    "BANK_ACCOUNT": re.compile(r"\b(?:account\s*(?:number|no\.?)?\s*[:\-]?\s*)\d{8,18}\b", re.IGNORECASE),
    "AADHAAR_LIKE": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
}

# Keywords that, combined with nearby numbers, strongly indicate a secret.
SENSITIVE_KEYWORDS = [
    "password", "pwd", "otp", "cvv", "pin", "ssn", "social security",
    "card number", "account number", "routing number", "security code",
]


def redact_text(text: str) -> tuple[str, list[str]]:
    """
    Returns (redacted_text, list_of_flagged_categories).
    Sensitive spans are replaced with [REDACTED:<CATEGORY>].
    """
    if not text:
        return text, []

    flagged = []
    redacted = text

    # Targeted phrase-level redaction first (password IS xyz123 -> whole phrase)
    for category, pattern in PATTERNS.items():
        def _sub(match, cat=category):
            flagged.append(cat)
            return f"[REDACTED:{cat}]"

        new_redacted = pattern.sub(_sub, redacted)
        redacted = new_redacted

    return redacted, sorted(set(flagged))


def contains_sensitive_keywords(text: str) -> list[str]:
    """Lightweight keyword scan used for warning banners in the UI."""
    lowered = text.lower()
    return [kw for kw in SENSITIVE_KEYWORDS if kw in lowered]


def safe_preview(text: str, max_chars: int = 4000) -> str:
    """Returns a redacted, length-capped version of text safe for display/export."""
    redacted, _ = redact_text(text)
    if len(redacted) > max_chars:
        return redacted[:max_chars] + "\n...[truncated for length]..."
    return redacted
