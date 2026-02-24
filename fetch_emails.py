"""Fetch the 100 most recent Gmail messages and save to JSON.

Output format (emails.json):
    [
        {
            "email_address": "sender@example.com",
            "subject": "...",
            "body": "..."
        },
        ...
    ]

Requirements:
    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

Setup:
    1. Create a project at https://console.cloud.google.com/
    2. Enable the Gmail API.
    3. Create OAuth 2.0 credentials (Desktop app) and download as credentials.json.
    4. Place credentials.json in the same directory as this script (or pass --credentials).
    5. Run: python fetch_emails.py
       The first run opens a browser for OAuth consent; token.json is saved for reuse.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Read-only scope — never modifies or deletes messages.
_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

_DEFAULT_CREDENTIALS = Path("credentials.json")
_DEFAULT_TOKEN = Path("token.json")
_DEFAULT_OUTPUT = Path("emails.json")
_DEFAULT_COUNT = 100


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_credentials(credentials_path: Path, token_path: Path) -> Credentials:
    """Return valid OAuth2 credentials, refreshing or re-authorising as needed."""
    creds: Credentials | None = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                sys.exit(
                    f"credentials.json not found at '{credentials_path}'.\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), _SCOPES
            )
            creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json())
        print(f"Token saved to {token_path}")

    return creds


# ---------------------------------------------------------------------------
# Body decoding helpers
# ---------------------------------------------------------------------------

def _b64_decode(data: str) -> bytes:
    """Decode Gmail's URL-safe base64 payload."""
    return base64.urlsafe_b64decode(data + "==")  # padding is always safe to add


def _strip_html(html: str) -> str:
    """Very lightweight HTML → plain-text: remove tags, collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_body(payload: dict[str, Any]) -> str:
    """Recursively extract the best available body text from a message payload.

    Preference order: text/plain > text/html > empty string.
    For multipart messages all text/plain parts are concatenated.
    """
    mime_type: str = payload.get("mimeType", "")
    parts: list[dict] = payload.get("parts", [])
    body_data: str = (payload.get("body") or {}).get("data", "")

    if not parts:
        # Leaf node
        if not body_data:
            return ""
        raw = _b64_decode(body_data).decode("utf-8", errors="replace")
        return _strip_html(raw) if "html" in mime_type else raw

    # multipart/* — prefer plain, fall back to html
    plain_parts: list[str] = []
    html_parts: list[str] = []

    for part in parts:
        part_mime: str = part.get("mimeType", "")
        part_body_data: str = (part.get("body") or {}).get("data", "")

        if part_mime == "text/plain" and part_body_data:
            plain_parts.append(
                _b64_decode(part_body_data).decode("utf-8", errors="replace")
            )
        elif part_mime == "text/html" and part_body_data:
            html_parts.append(
                _strip_html(_b64_decode(part_body_data).decode("utf-8", errors="replace"))
            )
        elif part_mime.startswith("multipart/"):
            # Recurse into nested multipart
            nested = _extract_body(part)
            if nested:
                plain_parts.append(nested)

    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        return "\n".join(html_parts).strip()
    return ""


def _extract_header(headers: list[dict[str, str]], name: str) -> str:
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def _extract_sender_address(from_header: str) -> str:
    """Return the bare email address from a 'From' header value.

    Handles both 'Name <addr>' and bare 'addr' formats.
    """
    match = re.search(r"<([^>]+)>", from_header)
    if match:
        return match.group(1).strip()
    return from_header.strip()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_emails(
    credentials_path: Path,
    token_path: Path,
    count: int,
) -> list[dict[str, str]]:
    """Return up to `count` most recent emails as a list of dicts."""
    creds = _get_credentials(credentials_path, token_path)
    service = build("gmail", "v1", credentials=creds)
    users = service.users()  # type: ignore[attr-defined]

    # 1. List message IDs (newest first)
    print(f"Fetching list of {count} message IDs…")
    message_ids: list[str] = []
    page_token: str | None = None

    while len(message_ids) < count:
        batch_size = min(count - len(message_ids), 500)  # API max per page is 500
        kwargs: dict[str, Any] = {"userId": "me", "maxResults": batch_size}
        if page_token:
            kwargs["pageToken"] = page_token
        response = users.messages().list(**kwargs).execute()
        messages = response.get("messages", [])
        message_ids.extend(m["id"] for m in messages)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    message_ids = message_ids[:count]
    total = len(message_ids)
    print(f"Found {total} messages. Downloading…")

    # 2. Fetch each message in full
    results: list[dict[str, str]] = []
    for i, msg_id in enumerate(message_ids, 1):
        msg = users.messages().get(
            userId="me", id=msg_id, format="full"
        ).execute()

        payload = msg.get("payload", {})
        headers: list[dict[str, str]] = payload.get("headers", [])

        from_raw = _extract_header(headers, "From")
        email_address = _extract_sender_address(from_raw)
        subject = _extract_header(headers, "Subject")
        body = _extract_body(payload)

        results.append(
            {
                "email_address": email_address,
                "subject": subject,
                "body": body,
            }
        )

        if i % 10 == 0 or i == total:
            print(f"  {i}/{total}", end="\r", flush=True)

    print()  # newline after progress
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download recent Gmail messages and save to JSON."
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=_DEFAULT_CREDENTIALS,
        help=f"Path to credentials.json (default: {_DEFAULT_CREDENTIALS})",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=_DEFAULT_TOKEN,
        help=f"Path to store/load token.json (default: {_DEFAULT_TOKEN})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=_DEFAULT_COUNT,
        help=f"Number of emails to fetch (default: {_DEFAULT_COUNT})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    emails = fetch_emails(
        credentials_path=args.credentials,
        token_path=args.token,
        count=args.count,
    )

    args.output.write_text(
        json.dumps(emails, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(emails)} emails to {args.output}")


if __name__ == "__main__":
    main()
