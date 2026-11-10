"""Contact normalisation, text fingerprints and PII redaction.

These helpers are deliberately dependency-free: the database migration, the
upload service and the audit logger all use the same rules, so a candidate
created before de-duplication existed still matches a new upload.
"""

from __future__ import annotations

import hashlib
import re

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# A phone written with at least ten digits, optionally grouped by spaces,
# dashes, dots or parentheses. The digit count is checked separately so years
# such as "2024" never become a phone number.
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")
PROFILE_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def normalize_email(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip().strip(".,;:<>()[]").lower()
    if not EMAIL_PATTERN.fullmatch(candidate):
        return None
    return candidate


def normalize_phone(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) < 10 or len(digits) > 15:
        return None
    # Russian numbers are commonly written with a leading 8 instead of +7.
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    return digits


def extract_email(text: str) -> str | None:
    match = EMAIL_PATTERN.search(text)
    return normalize_email(match.group(0)) if match else None


def extract_phone(text: str) -> str | None:
    for match in PHONE_PATTERN.finditer(text):
        phone = normalize_phone(match.group(0))
        if phone:
            return phone
    return None


def content_hash(text: str) -> str:
    """Stable fingerprint of resume text, ignoring spacing and letter case."""

    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def mask_email(value: str | None) -> str | None:
    """Keep an audit entry useful without storing the full address."""

    if not value or "@" not in value:
        return None
    local, _, domain = value.partition("@")
    return f"{local[:1]}***@{domain}"


def redact_resume_text(text: str, limit: int = 200) -> str:
    """Remove contacts and links before an excerpt could reach a log.

    New audit events do not store resume excerpts at all. This helper exists for
    diagnostics and keeps the rule in one place if an excerpt is ever needed.
    """

    redacted = EMAIL_PATTERN.sub("[email]", text)
    redacted = PHONE_PATTERN.sub("[phone]", redacted)
    redacted = PROFILE_URL_PATTERN.sub("[link]", redacted)
    return redacted[:limit]
