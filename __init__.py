"""email_classifier — internal Gmail module helper.

Not exposed to the LLM or module loader. Import directly from gmail code::

    from .email_classifier import EmailClassifier, EmailScores
"""

from .classifier import EmailClassifier, EmailScores
from .heuristics import N_HEURISTIC_FEATURES, compute_heuristic_vector

__all__ = [
    "EmailClassifier",
    "EmailScores",
    "compute_heuristic_vector",
    "N_HEURISTIC_FEATURES",
]
