"""Interactive inference — type email details, get classifier scores.

Usage:
    python infer.py                     # loads head.pt by default
    python infer.py --head my_head.pt   # custom checkpoint
    python infer.py --no-head           # run with untrained (random) head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from email_classifier import EmailClassifier, EmailScores

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

def _ansi(code: str, t: str) -> str: return f"\033[{code}m{t}\033[0m"
def _bold(t: str)   -> str: return _ansi("1",  t)
def _dim(t: str)    -> str: return _ansi("2",  t)
def _cyan(t: str)   -> str: return _ansi("96", t)
def _yellow(t: str) -> str: return _ansi("93", t)
def _green(t: str)  -> str: return _ansi("92", t)
def _red(t: str)    -> str: return _ansi("91", t)
def _grey(t: str)   -> str: return _ansi("90", t)

_THRESHOLD = 0.5

_ROWS: list[tuple[str, str]] = [
    ("is_automated",       "Automated / bulk"),
    ("is_human_to_me",     "Human to me     "),
    ("is_time_sensitive",  "Time-sensitive  "),
    ("is_unpleasant",      "Unpleasant       "),
    ("needs_notification", "Notify me        "),
]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    bar = "█" * filled + _dim("░" * (width - filled))
    return bar


def _print_scores(scores: EmailScores) -> None:
    print()
    print(_bold("  Results:"))
    print(_grey("  " + "─" * 52))
    for field, label in _ROWS:
        score: float = getattr(scores, field)
        flag = score >= _THRESHOLD
        colour = _green if flag else _red
        indicator = colour(_bold("YES")) if flag else colour("no ")
        bar = _bar(score)
        print(f"  {label}  {bar}  {score:.2f}  {indicator}")
    print(_grey("  " + "─" * 52))
    print()


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _prompt(label: str, hint: str = "") -> str:
    suffix = _grey(f"  ({hint})") if hint else ""
    try:
        return input(f"  {_bold(label)}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _prompt_body() -> str:
    """Read a multi-line body. Blank line on its own ends input."""
    print(f"  {_bold('Body')} {_grey('(blank line to finish)')}")
    lines: list[str] = []
    try:
        while True:
            line = input("  > ")
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        print()
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(clf: EmailClassifier) -> None:
    print(_bold("\nEmail classifier — interactive inference"))
    print(_grey("Press Ctrl+C or leave all fields blank to exit.\n"))

    while True:
        print(_grey("─" * 56))
        sender  = _prompt("From   ", "e.g. alice@gmail.com")
        subject = _prompt("Subject", "e.g. Meeting tomorrow")
        body    = _prompt_body()

        if not sender and not subject and not body:
            print(_grey("Nothing entered — exiting."))
            break

        scores = clf.classify(sender, subject, body)
        _print_scores(scores)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if sys.platform == "win32":
        import os; os.system("")  # enable ANSI in Windows terminal

    parser = argparse.ArgumentParser(description="Interactive email classifier inference.")
    parser.add_argument(
        "--head", type=Path, default=Path("head.pt"),
        help="Path to trained head checkpoint (default: head.pt)",
    )
    parser.add_argument(
        "--no-head", action="store_true",
        help="Skip loading a checkpoint — use untrained (random) head weights",
    )
    args = parser.parse_args()

    print("Loading BERT…", end=" ", flush=True)
    if args.no_head:
        clf = EmailClassifier()
        print("done (no trained head loaded)")
    else:
        if not args.head.exists():
            sys.exit(
                f"Checkpoint not found: {args.head}\n"
                "Run train.py first, or use --no-head to test without trained weights."
            )
        clf = EmailClassifier.load(args.head)
        print(f"done  [{args.head}]")

    run(clf)


if __name__ == "__main__":
    main()
