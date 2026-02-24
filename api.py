"""High-level API wrapper around EmailClassifier.

Intended for use by higher-level modules that don't need to know about
BERT, heuristics, or checkpoint loading details.

Example::

    from email_classifier.api import EmailAPI

    api = EmailAPI()                        # loads head.pt by default
    result = api.infer("alice@gmail.com", "Lunch tomorrow?", "Hi, are you free?")

    if result["needs_notification"]:
        notify(...)

    print(result["scores"])                 # raw floats dict
    print(result["needs_notification"])     # bool
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from email_classifier import EmailClassifier, EmailScores


class EmailAPI:
    """Thin wrapper around EmailClassifier for use by higher-level modules.

    Args:
        head_path: Path to a trained head checkpoint saved by train.py.
                   Pass None to use untrained (random) weights.
    """

    def __init__(self, head_path: str | Path | None = Path("head.pt")) -> None:
        if head_path is not None:
            self._clf = EmailClassifier.load(head_path)
        else:
            self._clf = EmailClassifier()

    def infer(self, sender: str, subject: str, body: str) -> dict:
        """Classify an email and return scores and boolean flags.

        Args:
            sender:  Sender email address, e.g. "alice@gmail.com".
            subject: Email subject line.
            body:    Email body text (plain text or HTML).

        Returns:
            dict with two keys:

            ``scores`` — raw float values (0.0–1.0) for each label::

                {
                    "is_automated":       0.92,
                    "is_human_to_me":     0.03,
                    "is_time_sensitive":  0.11,
                    "is_unpleasant":      0.05,
                    "needs_notification": 0.08,
                }

            Each top-level label key — boolean shortcut (score >= 0.5)::

                result["needs_notification"]  # True / False
                result["is_automated"]        # True / False
        """
        scores: EmailScores = self._clf.classify(sender, subject, body)
        raw: dict[str, float] = asdict(scores)

        result: dict = {"scores": raw}
        for field, value in raw.items():
            result[field] = value >= 0.5

        return result
