"""Train the EmailClassifierHead on labelled entries in emails.json.

BERT weights are frozen — only the 3-layer head (~200K params) is trained.
All BERT embeddings are pre-computed once and cached in memory so that
subsequent epochs are fast even on CPU.

Usage:
    python train.py
    python train.py --data emails.json --epochs 20 --output head.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running as a standalone script inside the package directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from email_classifier import EmailClassifier
from email_classifier.heuristics import compute_heuristic_vector

_LABEL_FIELDS = [
    "is_automated",
    "is_human_to_me",
    "is_time_sensitive",
    "is_unpleasant",
    "needs_notification",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_labelled(path: Path) -> list[dict]:
    all_emails: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    labelled = [e for e in all_emails if "labels" in e]
    skipped = len(all_emails) - len(labelled)
    print(f"Loaded {len(labelled)} labelled emails "
          f"(skipped {skipped} without labels).")
    if not labelled:
        sys.exit("No labelled emails found — run label_emails.py first.")
    return labelled


# ---------------------------------------------------------------------------
# Feature pre-computation
# ---------------------------------------------------------------------------

def _build_tensors(
    clf: EmailClassifier,
    emails: list[dict],
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, Y) where X is (N, 780) feature matrix and Y is (N, 5) targets."""
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    total = len(emails)

    for i, email in enumerate(emails, 1):
        addr    = email.get("email_address", "")
        subject = email.get("subject", "")
        body    = email.get("body", "")

        bert_emb = clf._get_bert_embedding(subject, body)           # (768,)
        h = compute_heuristic_vector(addr, subject, body)
        h_t = torch.tensor(h, dtype=torch.float32, device=clf._device)  # (12,)
        xs.append(torch.cat([bert_emb, h_t]))                       # (780,)

        targets = [float(email["labels"][f]) for f in _LABEL_FIELDS]
        ys.append(torch.tensor(targets, dtype=torch.float32))

        if i % 10 == 0 or i == total:
            print(f"  [{label}] embedding {i}/{total}…", end="\r", flush=True)

    print()
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_path: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    seed: int,
) -> None:
    emails = _load_labelled(data_path)

    if len(emails) < 4:
        sys.exit("Need at least 4 labelled emails to train.")

    random.seed(seed)
    random.shuffle(emails)

    n_val = max(1, int(len(emails) * val_split))
    val_emails   = emails[:n_val]
    train_emails = emails[n_val:]
    print(f"Split: {len(train_emails)} train / {n_val} val\n")

    print("Loading BERT (frozen)…")
    clf = EmailClassifier()

    print(f"\nPre-computing embeddings for {len(train_emails)} training emails…")
    X_train, Y_train = _build_tensors(clf, train_emails, "train")

    print(f"Pre-computing embeddings for {len(val_emails)} validation emails…")
    X_val, Y_val = _build_tensors(clf, val_emails, "val")

    # Move everything to the target device
    device = clf._device
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val,   Y_val   = X_val.to(device),   Y_val.to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(clf._head.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_state: dict | None = None

    print(f"\nTraining for {epochs} epochs  "
          f"(lr={lr}, batch={batch_size}, device={device})\n")

    for epoch in range(1, epochs + 1):
        # --- train ---
        clf._head.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(clf._head(X_batch), Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X_batch)
        train_loss = running_loss / len(X_train)

        # --- validate ---
        clf._head.eval()
        with torch.no_grad():
            val_loss = criterion(clf._head(X_val), Y_val).item()

        marker = "  ← best" if val_loss < best_val_loss else ""
        print(f"  epoch {epoch:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in clf._head.state_dict().items()}

    # Restore best checkpoint and save
    if best_state is not None:
        clf._head.load_state_dict(best_state)

    clf.save(output_path)
    print(f"\nSaved best head (val_loss={best_val_loss:.4f}) → {output_path}")


# ---------------------------------------------------------------------------
# Per-label accuracy report on the full labelled set
# ---------------------------------------------------------------------------

def _accuracy_report(clf: EmailClassifier, emails: list[dict]) -> None:
    print("\nAccuracy on full labelled set (threshold = 0.5):")
    device = clf._device
    clf._head.eval()

    correct = [0] * 5
    total = len(emails)

    for email in emails:
        addr    = email.get("email_address", "")
        subject = email.get("subject", "")
        body    = email.get("body", "")
        bert_emb = clf._get_bert_embedding(subject, body)
        h_t = torch.tensor(
            compute_heuristic_vector(addr, subject, body),
            dtype=torch.float32, device=device,
        )
        x = torch.cat([bert_emb, h_t]).unsqueeze(0)
        with torch.no_grad():
            preds = clf._head(x).squeeze(0).tolist()
        for j, field in enumerate(_LABEL_FIELDS):
            pred_label = int(preds[j] >= 0.5)
            true_label = int(email["labels"][field])
            if pred_label == true_label:
                correct[j] += 1

    for j, field in enumerate(_LABEL_FIELDS):
        pct = 100.0 * correct[j] / total
        print(f"  {field:<22s}  {correct[j]:3d}/{total}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the EmailClassifier head on labelled emails."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("emails.json"),
        help="Labelled emails JSON (default: emails.json)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("head.pt"),
        help="Output checkpoint path (default: head.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of data held out for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffle/split (default: 42)",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip the per-label accuracy report after training",
    )
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )

    if not args.no_report:
        # Reload best weights and run accuracy on the full set
        clf = EmailClassifier.load(args.output)
        emails = _load_labelled(args.data)
        _accuracy_report(clf, emails)


if __name__ == "__main__":
    main()
