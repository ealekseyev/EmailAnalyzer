"""PyTorch BERT email classifier with multi-label output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from .heuristics import N_HEURISTIC_FEATURES, compute_heuristic_vector

_BERT_HIDDEN: int = 768
_HEAD_INPUT_DIM: int = _BERT_HIDDEN + N_HEURISTIC_FEATURES  # 780


@dataclass
class EmailScores:
    """Five independent binary classification scores, each in [0.0, 1.0].

    Scores are produced by sigmoid activations — NOT softmax. Each score
    is independent and can be thresholded separately (default: >= 0.5).
    """

    is_automated: float
    """1 = automated/bulk email, 0 = not automated."""

    is_human_to_me: float
    """1 = personally written by a human addressed to me, 0 = not."""

    is_time_sensitive: float
    """1 = time-sensitive regardless of source, 0 = not urgent."""

    is_unpleasant: float
    """1 = unpleasant or negative in tone, 0 = pleasant or neutral."""

    needs_notification: float
    """1 = I should be notified about this email, 0 = can be ignored."""


class EmailClassifierHead(nn.Module):
    """Three-layer linear head operating on concatenated BERT + heuristic features.

    Architecture:
        Linear(input_dim, 256) -> LayerNorm -> GELU -> Dropout(0.3)
        Linear(256, 64)        -> LayerNorm -> GELU -> Dropout(0.2)
        Linear(64, 5)          -> Sigmoid (element-wise, not softmax)
    """

    def __init__(self, input_dim: int = _HEAD_INPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmailClassifier:
    """Combines frozen pretrained BERT with a trainable multi-label classification head.

    BERT produces a CLS token embedding (768-dim) for the email subject+body.
    A 12-dim heuristic feature vector is concatenated and fed through a
    3-layer linear head yielding 5 independent binary scores via sigmoid.

    BERT weights are frozen — only the head (~200K params) is trained.
    This makes fine-tuning feasible on a small personal email dataset
    without GPU resources.

    Usage::

        clf = EmailClassifier()
        scores = clf.classify("sender@example.com", "Meeting tomorrow", "Hi...")
        if scores.needs_notification >= 0.5:
            await messenger.notify(...)

        clf.save("head.pt")
        clf2 = EmailClassifier.load("head.pt")
    """

    _BERT_MODEL_NAME: str = "bert-base-uncased"
    _MAX_LENGTH: int = 512

    def __init__(self, device: str | None = None) -> None:
        self._device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: BertTokenizer = BertTokenizer.from_pretrained(self._BERT_MODEL_NAME)
        self._bert: BertModel = BertModel.from_pretrained(self._BERT_MODEL_NAME)
        self._bert.eval()
        self._bert.to(self._device)
        self._head = EmailClassifierHead(input_dim=_BERT_HIDDEN + N_HEURISTIC_FEATURES)
        self._head.to(self._device)
        self._head.eval()

    def _get_bert_embedding(self, subject: str, body: str) -> torch.Tensor:
        """Return the CLS token embedding for subject+body, shape (768,)."""
        if len(body) > 10_000:
            body = body[:10_000]
        inputs = self._tokenizer(
            text=subject or "(no subject)",
            text_pair=body or "(empty body)",
            return_tensors="pt",
            truncation=True,
            max_length=self._MAX_LENGTH,
            padding=False,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._bert(**inputs)
        # CLS token is the first token of last_hidden_state
        return output.last_hidden_state[:, 0, :].squeeze(0)  # (768,)

    def classify(
        self,
        email_address: str,
        subject: str,
        body: str,
    ) -> EmailScores:
        """Classify an email and return five independent binary scores.

        Args:
            email_address: Sender address, e.g. "user@example.com".
            subject: Email subject line.
            body: Full email body text (plain text or HTML).

        Returns:
            EmailScores with five floats each in [0.0, 1.0].
        """
        subject = (subject or "").strip()
        body = (body or "").strip()
        email_address = (email_address or "").strip()

        bert_emb = self._get_bert_embedding(subject, body)  # (768,)

        heuristics = compute_heuristic_vector(email_address, subject, body)
        h_tensor = torch.tensor(heuristics, dtype=torch.float32, device=self._device)  # (12,)

        x = torch.cat([bert_emb, h_tensor], dim=0).unsqueeze(0)  # (1, 780)

        with torch.no_grad():
            scores = self._head(x)  # (1, 5)

        s = scores.squeeze(0).tolist()  # [float x5]
        return EmailScores(
            is_automated=s[0],
            is_human_to_me=s[1],
            is_time_sensitive=s[2],
            is_unpleasant=s[3],
            needs_notification=s[4],
        )

    def save(self, path: str | Path) -> None:
        """Save the head weights to disk. BERT is not saved (re-downloaded on demand).

        The checkpoint stores the head state_dict plus the input_dim so that
        load() can reconstruct the correct architecture even if N_HEURISTIC_FEATURES
        changes in a future version.
        """
        torch.save(
            {
                "state_dict": self._head.state_dict(),
                "input_dim": _BERT_HIDDEN + N_HEURISTIC_FEATURES,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "EmailClassifier":
        """Load a classifier from a saved head checkpoint.

        Args:
            path: Path to a checkpoint saved by save().
            device: Target device. Auto-detected if None.

        Returns:
            EmailClassifier with head weights loaded and in eval mode.
        """
        instance = cls(device=device)
        # weights_only=True avoids arbitrary pickle execution on deserialization
        ckpt = torch.load(Path(path), map_location=instance._device, weights_only=True)
        saved_input_dim: int = ckpt.get("input_dim", _BERT_HIDDEN + N_HEURISTIC_FEATURES)
        if saved_input_dim != _BERT_HIDDEN + N_HEURISTIC_FEATURES:
            # Checkpoint was saved with a different N_HEURISTIC_FEATURES — rebuild head
            instance._head = EmailClassifierHead(input_dim=saved_input_dim).to(instance._device)
        instance._head.load_state_dict(ckpt["state_dict"])
        instance._head.eval()
        return instance
