"""Interactive terminal tool for labelling emails as training data.

Reads emails.json, shows each unlabelled email, prompts for 5 binary labels,
and writes them back into the same file under a "labels" key.

Usage:
    python label_emails.py
    python label_emails.py --file emails.json --preview 300

Controls:
    Enter 0 or 1 for each prompt.
    Press Ctrl+C at any time — progress is saved after every email.
    Type 's' to skip an email (leaves it unlabelled for later).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade gracefully on Windows without colour support)
# ---------------------------------------------------------------------------

def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def _bold(t: str) -> str:   return _ansi("1", t)
def _dim(t: str) -> str:    return _ansi("2", t)
def _cyan(t: str) -> str:   return _ansi("96", t)
def _yellow(t: str) -> str: return _ansi("93", t)
def _green(t: str) -> str:  return _ansi("92", t)
def _red(t: str) -> str:    return _ansi("91", t)
def _grey(t: str) -> str:   return _ansi("90", t)

# ---------------------------------------------------------------------------
# Auto-classification rules
# Keyed by substring to match in email_address (case-insensitive).
# Value is the labels dict to apply without prompting.
# ---------------------------------------------------------------------------

_AUTO_RULES: list[tuple[str, dict[str, int]]] = [
    ("shein", {"is_automated": 1, "is_human_to_me": 0, "is_time_sensitive": 0, "is_unpleasant": 0, "needs_notification": 0}),
    ("ikon",  {"is_automated": 1, "is_human_to_me": 0, "is_time_sensitive": 0, "is_unpleasant": 0, "needs_notification": 0}),
]


def _auto_classify(email: dict) -> dict[str, int] | None:
    """Return auto labels if any rule matches the sender address, else None."""
    addr = email.get("email_address", "").lower()
    for substring, labels in _AUTO_RULES:
        if substring in addr:
            return labels
    return None


# ---------------------------------------------------------------------------
# The 5 label fields, in the same order as EmailScores
# ---------------------------------------------------------------------------

LABELS: list[tuple[str, str]] = [
    ("is_automated",      "Automated / bulk mail?         "),
    ("is_human_to_me",    "Personally written to you?     "),
    ("is_time_sensitive", "Time-sensitive?                "),
    ("is_unpleasant",     "Unpleasant / negative tone?    "),
    ("needs_notification","Should notify you?             "),
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _clear_line() -> None:
    print("\033[2K\r", end="")


def _divider(char: str = "─", width: int = 72) -> None:
    print(_grey(char * width))


def _show_email(email: dict, index: int, total: int, preview_chars: int) -> None:
    _divider()
    progress = _grey(f"[{index}/{total}]")
    already = _green(" ✓ already labelled") if "labels" in email else ""
    print(f"{progress}{already}")

    sender  = email.get("email_address", "(unknown sender)")
    subject = email.get("subject", "(no subject)")
    body    = email.get("body", "")

    print(f"  {_bold('From:')}    {_cyan(sender)}")
    print(f"  {_bold('Subject:')} {_yellow(subject)}")

    preview = body[:preview_chars].strip()
    if len(body) > preview_chars:
        preview += _grey("…")
    if preview:
        print()
        for line in preview.splitlines()[:6]:   # at most 6 lines
            print(f"  {_dim(line)}")
    print()


def _prompt_label(name: str, description: str) -> int | None:
    """Ask for a single 0/1 answer. Returns None if user skips.

    Enter = 0, '1' = 1, 's' = skip email.
    """
    while True:
        try:
            raw = input(f"  {description}{_bold('[Enter=0 / 1 / s]')} ").strip().lower()
        except EOFError:
            return None
        if raw in ("", "0"):
            return 0
        if raw == "1":
            return 1
        if raw == "s":
            return None
        print(f"  {_red('→ please enter 1, or just press Enter for 0, or s to skip')}")


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def label_emails(path: Path, preview_chars: int, relabel: bool) -> None:
    if not path.exists():
        sys.exit(f"File not found: {path}")

    emails: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    total = len(emails)

    # Figure out which need labelling
    to_label = [
        i for i, e in enumerate(emails)
        if relabel or "labels" not in e
    ]
    pending = len(to_label)

    if pending == 0:
        print(_green("All emails are already labelled. Use --relabel to redo them."))
        return

    done_before = total - pending
    print(_bold(f"\nEmail labeller — {path}"))
    print(f"{done_before} already labelled, {pending} remaining.\n")
    print(_grey("Enter 0 or 1 for each question, or 's' to skip the email."))
    print(_grey("Ctrl+C saves progress and exits.\n"))

    labelled_this_session = 0

    try:
        for rank, idx in enumerate(to_label, 1):
            email = emails[idx]
            display_index = done_before + rank

            _show_email(email, display_index, total, preview_chars)

            auto = _auto_classify(email)
            if auto is not None:
                email["labels"] = auto
                labelled_this_session += 1
                path.write_text(
                    json.dumps(emails, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(_grey(f"  auto-classified: {auto}\n"))
                continue

            collected: dict[str, int] = {}
            skip = False

            for field, description in LABELS:
                answer = _prompt_label(field, description)
                if answer is None:
                    skip = True
                    break
                collected[field] = answer

            if skip:
                print(_grey("  skipped\n"))
                continue

            email["labels"] = collected
            labelled_this_session += 1

            # Save immediately after every email
            path.write_text(
                json.dumps(emails, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(_green(f"  saved ✓\n"))

    except KeyboardInterrupt:
        print(f"\n\n{_yellow('Interrupted.')} Labelled {labelled_this_session} email(s) this session.")
    else:
        total_labelled = sum(1 for e in emails if "labels" in e)
        print(_green(f"\nDone! {labelled_this_session} labelled this session, {total_labelled}/{total} total."))


# ---------------------------------------------------------------------------
# Human-email fetcher
# ---------------------------------------------------------------------------

# Personal / institutional domains considered "human" senders.
# Mirrors the trusted set in heuristics.py.
_HUMAN_DOMAINS: list[str] = [
    "gmail.com",
    "sjsu.edu",
    "outlook.com",
    "hotmail.com",
    "yahoo.com",
    "icloud.com",
    "me.com",
    "protonmail.com",
    "pm.me",
]

_GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _get_credentials(credentials_path: Path, token_path: Path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                sys.exit(
                    f"credentials.json not found at '{credentials_path}'.\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), _GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
        print(f"Token saved to {token_path}")
    return creds


def _b64_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "==")


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    for ent, repl in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")):
        text = text.replace(ent, repl)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_body(payload: dict[str, Any]) -> str:
    mime_type: str = payload.get("mimeType", "")
    parts: list[dict] = payload.get("parts", [])
    body_data: str = (payload.get("body") or {}).get("data", "")
    if not parts:
        if not body_data:
            return ""
        raw = _b64_decode(body_data).decode("utf-8", errors="replace")
        return _strip_html(raw) if "html" in mime_type else raw
    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in parts:
        part_mime: str = part.get("mimeType", "")
        part_data: str = (part.get("body") or {}).get("data", "")
        if part_mime == "text/plain" and part_data:
            plain_parts.append(_b64_decode(part_data).decode("utf-8", errors="replace"))
        elif part_mime == "text/html" and part_data:
            html_parts.append(_strip_html(_b64_decode(part_data).decode("utf-8", errors="replace")))
        elif part_mime.startswith("multipart/"):
            nested = _extract_body(part)
            if nested:
                plain_parts.append(nested)
    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        return "\n".join(html_parts).strip()
    return ""


def _header(headers: list[dict[str, str]], name: str) -> str:
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def _bare_address(from_header: str) -> str:
    m = re.search(r"<([^>]+)>", from_header)
    return m.group(1).strip() if m else from_header.strip()


def fetch_human_emails(
    output_path: Path,
    credentials_path: Path = Path("credentials.json"),
    token_path: Path = Path("token.json"),
    count: int = 100,
) -> None:
    """Fetch up to `count` emails from personal/institutional domains and append to output_path.

    Uses a Gmail search query restricted to common human-sender domains
    (gmail.com, sjsu.edu, outlook.com, etc.) so the downloaded batch is
    predominantly person-to-person mail rather than automated bulk mail.

    Already-seen message IDs (stored in the JSON as "message_id") are
    skipped to avoid duplicates across multiple fetches.
    """
    from googleapiclient.discovery import build

    # Load existing emails and build dedup sets
    existing: list[dict] = []
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    # Primary dedup: by Gmail message ID (entries fetched by this tool)
    seen_ids: set[str] = {e["message_id"] for e in existing if "message_id" in e}
    # Secondary dedup: by (sender, subject) for entries without a message_id
    seen_pairs: set[tuple[str, str]] = {
        (e.get("email_address", "").lower(), e.get("subject", "").lower())
        for e in existing
    }

    creds = _get_credentials(credentials_path, token_path)
    service = build("gmail", "v1", credentials=creds)
    users = service.users()  # type: ignore[attr-defined]

    # Restrict to emails addressed TO this account from personal domains
    domain_clause = " OR ".join(f"from:@{d}" for d in _HUMAN_DOMAINS)
    query = f"to:giantsmilodon@gmail.com ({domain_clause})"

    print(f"Searching for emails to giantsmilodon@gmail.com from personal domains…")

    # Collect enough candidate IDs (fetch extra to account for already-seen ones)
    candidate_ids: list[str] = []
    page_token: str | None = None
    target = count + len(seen_ids) + 50  # fetch a buffer

    while len(candidate_ids) < target:
        batch = min(target - len(candidate_ids), 500)
        kwargs: dict[str, Any] = {"userId": "me", "maxResults": batch, "q": query}
        if page_token:
            kwargs["pageToken"] = page_token
        response = users.messages().list(**kwargs).execute()
        msgs = response.get("messages", [])
        candidate_ids.extend(m["id"] for m in msgs)
        page_token = response.get("nextPageToken")
        if not page_token or not msgs:
            break

    # Filter already-seen, then cap at count
    new_ids = [mid for mid in candidate_ids if mid not in seen_ids][:count]
    total = len(new_ids)

    if total == 0:
        print(_yellow("No new human emails found."))
        return

    print(f"Found {total} new messages. Downloading…")

    added = 0
    for i, msg_id in enumerate(new_ids, 1):
        msg = users.messages().get(userId="me", id=msg_id, format="full").execute()
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])

        from_raw = _header(headers, "From")
        email_address = _bare_address(from_raw)
        subject = _header(headers, "Subject")

        # Secondary dedup check (catches entries from fetch_emails.py without message_id)
        pair = (email_address.lower(), subject.lower())
        if pair in seen_pairs:
            continue

        entry: dict[str, Any] = {
            "message_id":    msg_id,
            "email_address": email_address,
            "subject":       subject,
            "body":          _extract_body(payload),
        }
        existing.append(entry)
        seen_pairs.add(pair)
        added += 1

        if i % 10 == 0 or i == total:
            print(f"  {i}/{total}", end="\r", flush=True)

    print()
    output_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_green(f"Added {added} human emails → {output_path}  (total: {len(existing)})"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # Enable ANSI on Windows
    if sys.platform == "win32":
        import os
        os.system("")  # activates VT100 mode in Windows terminal

    parser = argparse.ArgumentParser(description="Interactively label emails for classifier training.")
    parser.add_argument(
        "--file", type=Path, default=Path("emails.json"),
        help="Path to emails JSON file (default: emails.json)",
    )
    parser.add_argument(
        "--preview", type=int, default=400,
        help="Characters of body to preview (default: 400)",
    )
    parser.add_argument(
        "--relabel", action="store_true",
        help="Re-prompt emails that already have labels",
    )
    parser.add_argument(
        "--fetch-human", action="store_true",
        help="Fetch 100 emails from personal domains and append to --file, then exit",
    )
    parser.add_argument(
        "--credentials", type=Path, default=Path("credentials.json"),
        help="OAuth credentials file (default: credentials.json)",
    )
    parser.add_argument(
        "--token", type=Path, default=Path("token.json"),
        help="OAuth token cache file (default: token.json)",
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of human emails to fetch (default: 100)",
    )
    args = parser.parse_args()

    if args.fetch_human:
        fetch_human_emails(
            output_path=args.file,
            credentials_path=args.credentials,
            token_path=args.token,
            count=args.count,
        )
        return

    label_emails(args.file, args.preview, args.relabel)


if __name__ == "__main__":
    main()
