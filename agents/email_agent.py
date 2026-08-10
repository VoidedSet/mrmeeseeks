"""
email_agent.py — Mr Meeseeks Email Agent

Fetches emails from Gmail via IMAP, stores them in ChromaDB (meeseeks_emails collection)
with rich metadata. Registers IPC bus tools for the Brain to use.

Setup:
 - Enable IMAP in Gmail Settings > Forwarding and POP/IMAP
 - Enable 2FA on Google Account > Security > App passwords > generate 16-char password
 - Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env
"""
import imaplib
import email
import email.utils
import asyncio
import logging
import os
import re
from datetime import datetime
from email.header import decode_header
from typing import Optional

from core.ipc_bus import bus

log = logging.getLogger("email_agent")

# ─ Promotional sender/subject heuristics ─────────────────────────────
PROMO_KEYWORDS = [
    "unsubscribe", "click here", "special offer", "limited time",
    "% off", "deal", "sale", "discount", "coupon", "promo",
    "newsletter", "no-reply", "noreply", "donotreply", "marketing",
    "notifications@", "info@", "hello@", "team@", "support@",
]


def _decode_header_value(value: str) -> str:
    """Decode RFC2047 encoded header value to unicode string."""
    if not value:
        return ""
    parts = decode_header(value)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="ignore"))
        else:
            decoded.append(str(part))
    return "".join(decoded)


def _is_promotional(sender: str, subject: str, body_snippet: str) -> bool:
    """Heuristic to detect promotional/spam emails."""
    text = (sender + " " + subject + " " + body_snippet).lower()
    matches = sum(1 for kw in PROMO_KEYWORDS if kw in text)
    return matches >= 2  # require 2+ signals to avoid false positives


def _extract_body(msg) -> str:
    """Extract plain text body from email message."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in disposition:
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                        break
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")
        except Exception:
            pass
    # Collapse excessive whitespace
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()[:4000]  # cap at 4000 chars


def _get_sender_name(sender: str) -> str:
    """Extract display name from 'Name <email@domain>' format."""
    name, addr = email.utils.parseaddr(sender)
    return name if name else addr.split("@")[0]


class EmailAgent:
    """Gmail IMAP email fetcher with ChromaDB caching."""

    def __init__(self):
        self.gmail_address = os.environ.get("GMAIL_ADDRESS", "").strip()
        # Remove any spaces from app password (e.g. "abcd efgh ijkl mnop" -> "abcdefghijklmnop")
        raw_pw = os.environ.get("GMAIL_APP_PASSWORD", "")
        self.app_password = raw_pw.replace(" ", "").strip()
        self.imap_host = os.environ.get("IMAP_HOST", "imap.gmail.com")
        self.imap_port = int(os.environ.get("IMAP_PORT", "993"))
        self._mail: Optional[imaplib.IMAP4_SSL] = None

    def _connect(self) -> imaplib.IMAP4_SSL:
        """Create a fresh IMAP connection with 10-second socket timeout."""
        mail = imaplib.IMAP4_SSL(self.imap_host, self.imap_port, timeout=10.0)
        mail.login(self.gmail_address, self.app_password)
        return mail

    def _fetch_emails_sync(self, max_emails: int = 20, only_unread: bool = True) -> list[dict]:
        """
        Synchronous IMAP fetch. Returns list of email dicts.
        Called from asyncio.to_thread.
        """
        if not self.gmail_address or not self.app_password or "your@gmail.com" in self.gmail_address:
            log.warning("Email credentials not configured properly in .env (GMAIL_ADDRESS / GMAIL_APP_PASSWORD placeholder or missing)")
            return []

        emails = []
        try:
            mail = self._connect()
            mail.select("INBOX")

            search_criteria = "UNSEEN" if only_unread else "ALL"
            status, messages = mail.search(None, search_criteria)
            if status != "OK":
                return []

            email_ids = messages[0].split()
            # Fetch most recent first (reverse order)
            email_ids = list(reversed(email_ids[-max_emails:]))

            for e_id in email_ids:
                try:
                    _, msg_data = mail.fetch(e_id, "(RFC822)")
                    for response_part in msg_data:
                        if not isinstance(response_part, tuple):
                            continue
                        msg = email.message_from_bytes(response_part[1])

                        sender_raw = msg.get("From", "")
                        subject = _decode_header_value(msg.get("Subject", "(no subject)"))
                        date_str = msg.get("Date", "")
                        msg_id = msg.get("Message-ID", e_id.decode())

                        # Parse date
                        try:
                            parsed_date = email.utils.parsedate_to_datetime(date_str)
                            date_iso = parsed_date.strftime("%Y-%m-%d")
                            timestamp = int(parsed_date.timestamp())
                        except Exception:
                            date_iso = datetime.now().strftime("%Y-%m-%d")
                            timestamp = int(datetime.now().timestamp())

                        body = _extract_body(msg)
                        sender_name = _get_sender_name(sender_raw)
                        is_promo = _is_promotional(sender_raw, subject, body[:200])

                        emails.append({
                            "uid": e_id.decode(),
                            "message_id": msg_id,
                            "sender": sender_raw,
                            "sender_name": sender_name,
                            "subject": subject,
                            "date": date_iso,
                            "timestamp": timestamp,
                            "body": body,
                            "body_snippet": body[:300],
                            "is_read": False,
                            "is_promotional": is_promo,
                            "category": "promotional" if is_promo else "inbox",
                        })
                except Exception as e:
                    log.debug(f"Failed to parse email {e_id}: {e}")
                    continue

            mail.close()
            mail.logout()
        except Exception as e:
            log.error(f"IMAP fetch failed: {e}")

        return emails

    async def fetch_inbox(self, max_emails: int = 20, only_unread: bool = True) -> list[dict]:
        """Async IMAP fetch — runs in thread executor."""
        emails = await asyncio.to_thread(self._fetch_emails_sync, max_emails, only_unread)

        # Store to ChromaDB asynchronously (fire and forget)
        if emails:
            asyncio.create_task(self._store_to_chroma(emails))

        return emails

    async def _store_to_chroma(self, emails: list[dict]):
        """Store fetched emails to ChromaDB meeseeks_emails collection."""
        try:
            from core.chroma_store import chroma_store
            for em in emails:
                doc_text = f"From: {em['sender_name']}\nSubject: {em['subject']}\n\n{em['body_snippet']}"
                metadata = {
                    "sender": em["sender"],
                    "sender_name": em["sender_name"],
                    "subject": em["subject"],
                    "date": em["date"],
                    "timestamp": em["timestamp"],
                    "is_read": em["is_read"],
                    "is_promotional": em["is_promotional"],
                    "email_uid": em["uid"],
                    "category": em["category"],
                }
                await asyncio.to_thread(
                    chroma_store.add_email,
                    doc_text,
                    em["message_id"],
                    metadata
                )
        except Exception as e:
            log.debug(f"ChromaDB email store failed: {e}")

    async def get_summary(self, n: int = 5, include_promo: bool = False) -> list[dict]:
        """
        Get a brief summary of recent unread emails.
        Returns list of {sender_name, subject, date} dicts.
        Filters out promotional by default.
        """
        emails = await self.fetch_inbox(max_emails=30)
        filtered = [e for e in emails if include_promo or not e["is_promotional"]]
        return [{"sender_name": e["sender_name"], "subject": e["subject"], "date": e["date"]} for e in filtered[:n]]

    async def get_email_body(self, uid: str) -> Optional[dict]:
        """Fetch full body of a specific email by UID."""
        emails = await self.fetch_inbox(max_emails=50, only_unread=False)
        for em in emails:
            if em["uid"] == uid:
                return em
        return None

    async def search_by_sender(self, sender_query: str) -> list[dict]:
        """Search emails by sender name (from ChromaDB semantic search)."""
        try:
            from core.chroma_store import chroma_store
            results = await asyncio.to_thread(
                chroma_store.search_emails,
                sender_query,
                limit=10
            )
            return results
        except Exception as e:
            log.debug(f"Email search failed: {e}")
            return []


# ─ Singleton ─────────────────────────────────────────────────────
_email_agent: Optional[EmailAgent] = None


def get_email_agent() -> EmailAgent:
    global _email_agent
    if _email_agent is None:
        _email_agent = EmailAgent()
    return _email_agent


# ─ IPC Bus handlers ──────────────────────────────────────────────────
async def handle_fetch_inbox(args: dict) -> dict:
    """Refresh inbox from Gmail."""
    agent = get_email_agent()
    if not agent.gmail_address or "your@gmail.com" in agent.gmail_address:
        return {"error": "Gmail address is not configured. Please set GMAIL_ADDRESS in .env"}
    if not agent.app_password:
        return {"error": "Gmail app password is missing. Please set GMAIL_APP_PASSWORD in .env"}

    max_emails = args.get("max", 20)
    try:
        emails = await asyncio.wait_for(agent.fetch_inbox(max_emails=max_emails), timeout=10.0)
    except asyncio.TimeoutError:
        log.warning("IMAP fetch_inbox timed out after 10 seconds")
        return {"error": "Gmail connection timed out after 10 seconds."}
    except Exception as e:
        log.error(f"IMAP fetch_inbox error: {e}")
        return {"error": f"Failed to fetch emails: {e}"}

    if not emails:
        return {"total": 0, "non_promotional": 0, "promotional_filtered": 0, "emails": [], "status": "No emails returned"}

    real = [e for e in emails if not e["is_promotional"]]
    promo_count = len(emails) - len(real)

    clean_items = []
    summary_lines = []
    for i, e in enumerate(real[:8], 1):
        s_name = _decode_header_value(e["sender_name"])
        subj = _decode_header_value(e["subject"])
        clean_items.append({"uid": e["uid"], "sender_name": s_name, "subject": subj, "date": e["date"]})
        summary_lines.append(f"{i}. From {s_name}: '{subj}'")

    summary_text = f"Found {len(real)} recent emails:\n" + "\n".join(summary_lines)
    return {
        "summary_text": summary_text,
        "instructions": "Speak the summary naturally to the user in conversational sentences. Do NOT format as a Markdown table or list with 'Subject:' or 'Sender:' field labels.",
        "emails": clean_items
    }


async def handle_get_email_summary(args: dict) -> dict:
    """Get brief summaries of recent unread non-promotional emails."""
    agent = get_email_agent()
    if not agent.gmail_address or "your@gmail.com" in agent.gmail_address:
        return {"result": "Gmail address is not configured in .env"}
    if not agent.app_password:
        return {"result": "Gmail app password is missing in .env"}

    n = args.get("count", 5)
    summary = await agent.get_summary(n=n)
    if not summary:
        return {"result": "No unread non-promotional emails found."}

    lines = []
    for i, e in enumerate(summary, 1):
        s_name = _decode_header_value(e["sender_name"])
        subj = _decode_header_value(e["subject"])
        lines.append(f"{i}. From {s_name}: '{subj}'")

    return {
        "result": "Unread emails:\n" + "\n".join(lines),
        "instructions": "Speak the email titles naturally. Do NOT use markdown labels like Subject or Sender Name."
    }


async def handle_read_email(args: dict) -> dict:
    """Read full email content by UID."""
    uid = str(args.get("uid", "")).strip()
    if not uid:
        return {"error": "Missing 'uid' argument. Use get_email_summary first to get UIDs."}
    agent = get_email_agent()
    em = await agent.get_email_body(uid)
    if not em:
        return {"error": f"Email with UID {uid} not found."}
    return {
        "from": _decode_header_value(em["sender"]),
        "subject": _decode_header_value(em["subject"]),
        "date": em["date"],
        "body": em["body"]
    }


async def handle_search_emails(args: dict) -> dict:
    """Search emails by sender name or query."""
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Missing 'query'."}
    agent = get_email_agent()
    results = await agent.search_by_sender(query)
    if not results:
        # Fall back to live IMAP fetch filtered by sender
        emails = await agent.fetch_inbox(max_emails=50, only_unread=False)
        results = [e for e in emails if query.lower() in e["sender_name"].lower() or query.lower() in e["sender"].lower() or query.lower() in e["subject"].lower()]
    if not results:
        return {"result": f"No emails found matching '{query}'."}

    lines = []
    for i, e in enumerate(results[:10], 1):
        s_name = _decode_header_value(e.get("sender_name") or e.get("sender", ""))
        subj = _decode_header_value(e.get("subject", ""))
        lines.append(f"{i}. From {s_name}: '{subj}'")

    return {
        "result": "\n".join(lines),
        "count": len(results),
        "instructions": "Summarize these search results naturally to the user. Do NOT repeat raw field labels."
    }


def register():
    """Register email tools on the IPC bus."""
    bus.register("fetch_inbox", handle_fetch_inbox)
    bus.register("get_email_summary", handle_get_email_summary)
    bus.register("read_email", handle_read_email)
    bus.register("search_emails", handle_search_emails)
    log.info("Email agent registered ✓")
