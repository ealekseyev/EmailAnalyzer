"""Rule-based heuristic feature extraction for email classification."""

from __future__ import annotations

import re

N_HEURISTIC_FEATURES: int = 12


def _parse_address(email_address: str) -> tuple[str, str]:
    """Split email_address into (local_part, domain). Handles missing '@'."""
    email_address = email_address.strip()
    if not email_address:
        return "", ""
    if "@" not in email_address:
        return email_address.lower(), ""
    local, _, domain = email_address.rpartition("@")
    return local.lower(), domain.lower()


def _noreply_local(local: str) -> float:
    patterns = {"noreply", "no-reply", "no_reply", "donotreply", "do-not-reply", "donot-reply"}
    return 1.0 if any(p in local for p in patterns) else 0.0


def _marketing_local(local: str) -> float:
    patterns = {
        "marketing", "newsletter", "news", "updates", "promo", "promotions",
        "offers", "deals", "announce", "notification", "notifications",
        "alert", "alerts", "digest",
    }
    return 1.0 if any(p in local for p in patterns) else 0.0


def _trusted_personal_domain(domain: str) -> float:
    trusted = {
        "gmail.com", "outlook.com", "hotmail.com", "yahoo.com",
        "icloud.com", "me.com", "sjsu.edu", "protonmail.com", "pm.me",
    }
    return 1.0 if domain in trusted else 0.0


def _bulk_sender_subdomain(domain: str) -> float:
    patterns = [
        "em.", "em2.", "mail.", "email.", "mailer.", "bounce.", "reply.",
        "send.", "delivery.", "msg.", "mta.", "sg.",
        "sendgrid", "mailgun", "mailchimp", "amazonses", "postmarkapp",
        "sparkpostmail", "mandrillapp", "exacttarget", "salesforce",
        "marketo", "hubspot",
    ]
    return 1.0 if any(p in domain for p in patterns) else 0.0


def _high_numeric_ratio_local(local: str) -> float:
    if not local:
        return 0.0
    ratio = sum(c.isdigit() for c in local) / len(local)
    return 1.0 if ratio >= 0.4 else 0.0


def _has_unsubscribe(body: str) -> float:
    body_lower = body.lower()
    patterns = [
        "unsubscribe", "opt out", "opt-out", "manage preferences",
        "email preferences", "notification settings", "remove me",
    ]
    return 1.0 if any(p in body_lower for p in patterns) else 0.0


def _has_html_content(body: str) -> float:
    tags = [
        "<html", "<HTML", "<body", "<BODY", "<table", "<TABLE",
        "<div", "<DIV", "<p style=", "<span style=",
    ]
    return 1.0 if any(t in body for t in tags) else 0.0


def _promo_subject_keywords(subject: str) -> float:
    subject_lower = subject.lower()
    patterns = [
        "% off", "% discount", "sale", "deal", "limited time", "offer expires",
        "coupon", "promo", "free shipping", "buy now", "act now", "claim your",
        "exclusive", "flash sale", "clearance", "save up to", "special offer",
        "today only", "last chance",
    ]
    return 1.0 if any(p in subject_lower for p in patterns) else 0.0


def _tracking_pixel_body(body: str) -> float:
    if len(body) <= 100:
        return 0.0
    stripped = re.sub(r"<[^>]+>", " ", body)
    word_count = len(stripped.split())
    return 1.0 if word_count < 20 else 0.0


def _no_personal_greeting(body: str) -> float:
    if len(body.strip()) < 30:
        return 0.0
    excerpt = body[:200].lower()
    greetings = [
        "hi ", "hello ", "dear ", "hey ", "good morning",
        "good afternoon", "good evening", "thanks,", "thank you,",
    ]
    return 0.0 if any(g in excerpt for g in greetings) else 1.0


def _system_sender_local(local: str) -> float:
    patterns = {"bounce", "mailer-daemon", "postmaster", "daemon", "system", "automated", "robot"}
    return 1.0 if any(p in local for p in patterns) else 0.0


def _short_or_empty_body(body: str) -> float:
    return 1.0 if len(body.strip()) < 50 else 0.0


def compute_heuristic_vector(
    email_address: str,
    subject: str,
    body: str,
) -> list[float]:
    """Return a normalized feature vector of length N_HEURISTIC_FEATURES.

    Each element is 0.0 or 1.0 (binary flag). The vector encodes rule-based
    signals that complement the BERT CLS embedding in the classifier head.

    Args:
        email_address: Sender address, e.g. "noreply@example.com".
        subject: Email subject line.
        body: Full email body text (plain or HTML).

    Returns:
        List of 12 floats, each 0.0 or 1.0.
    """
    subject = subject or ""
    body = body or ""
    email_address = email_address or ""

    local, domain = _parse_address(email_address)

    features: list[tuple[str, object]] = [
        ("noreply_local",            lambda: _noreply_local(local)),
        ("marketing_local",          lambda: _marketing_local(local)),
        ("trusted_personal_domain",  lambda: _trusted_personal_domain(domain)),
        ("bulk_sender_subdomain",    lambda: _bulk_sender_subdomain(domain)),
        ("high_numeric_ratio_local", lambda: _high_numeric_ratio_local(local)),
        ("has_unsubscribe",          lambda: _has_unsubscribe(body)),
        ("has_html_content",         lambda: _has_html_content(body)),
        ("promo_subject_keywords",   lambda: _promo_subject_keywords(subject)),
        ("tracking_pixel_body",      lambda: _tracking_pixel_body(body)),
        ("no_personal_greeting",     lambda: _no_personal_greeting(body)),
        ("system_sender_local",      lambda: _system_sender_local(local)),
        ("short_or_empty_body",      lambda: _short_or_empty_body(body)),
    ]

    result: list[float] = []
    for _, fn in features:
        try:
            result.append(float(fn()))
        except Exception:
            result.append(0.0)

    return result
