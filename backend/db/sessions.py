"""
sessions.py — PostgreSQL-backed session and message persistence

Manages two things:
  1. Sessions  — each chat tab is one session (title, folders, timestamps)
  2. Messages  — every user question and assistant answer within a session

Schema lives in db/connection.py (init_db). This module is read/write only.

Key differences from the previous SQLite version:
  - Uses the shared connection pool (db_conn context manager)
  - %s placeholders instead of ?
  - JSONB columns for folders, sources, confidence — no manual json.dumps/loads
  - TIMESTAMPTZ columns — datetimes stored with timezone, returned as datetime objects
  - Soft delete kept (is_active flag)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from db.connection import db_conn

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """One chat session — maps to one tab in the UI."""
    id:         str
    title:      str               # auto-generated from first question
    folders:    list[str]         # folder scope for this session
    created_at: datetime
    updated_at: datetime          # timestamp of last message
    is_active:  bool = True       # False = deleted from sidebar


@dataclass
class Message:
    """One message in a session — either user question or assistant answer."""
    id:          str
    session_id:  str
    role:        str              # "user" or "assistant"
    content:     str              # full text of message
    sources:     list[dict]       # citations [{"filename": ..., "page": ..., "score": ...}]
    confidence:  dict             # {"level": "HIGH"|"MEDIUM"|"LOW", "reason": "..."}
    model_used:  str              # which Ollama model answered
    timestamp:   datetime


# ── Store ─────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    All database operations for sessions and messages.
    Uses the shared PostgreSQL connection pool — no setup needed.
    """

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(
        self,
        folders: list[str] | None = None,
        title: str = "New Chat",
    ) -> Session:
        """
        Create a new session. Called when user clicks '+ New Chat'.
        Returns the new Session with a fresh UUID.
        """
        now = _now()
        session = Session(
            id=str(uuid4()),
            title=title,
            folders=folders or [],
            created_at=now,
            updated_at=now,
        )
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (id, title, folders, created_at, updated_at, is_active)
                    VALUES (%s, %s, %s::jsonb, %s, %s, %s)
                    """,
                    (
                        session.id,
                        session.title,
                        _json(session.folders),
                        session.created_at,
                        session.updated_at,
                        session.is_active,
                    ),
                )
        logger.info(f"Created session {session.id[:8]}...")
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Fetch a single session by ID. Returns None if not found."""
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM sessions WHERE id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
        return _row_to_session(row) if row else None

    def list_sessions(self, include_inactive: bool = False) -> list[Session]:
        """
        Return all sessions for the sidebar history list.
        Ordered by most recently updated first.
        """
        if include_inactive:
            query = "SELECT * FROM sessions ORDER BY updated_at DESC"
            params: tuple = ()
        else:
            query = "SELECT * FROM sessions WHERE is_active = TRUE ORDER BY updated_at DESC"
            params = ()

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [_row_to_session(r) for r in rows]

    def update_session_title(self, session_id: str, title: str) -> None:
        """
        Auto-update the session title from the first question asked.
        Truncates to 60 chars to fit the sidebar.
        """
        short_title = title[:60] + "..." if len(title) > 60 else title
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET title = %s, updated_at = %s WHERE id = %s",
                    (short_title, _now(), session_id),
                )

    def update_session_folders(self, session_id: str, folders: list[str]) -> None:
        """Update the folder scope for a session."""
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET folders = %s::jsonb, updated_at = %s WHERE id = %s",
                    (_json(folders), _now(), session_id),
                )

    def delete_session(self, session_id: str) -> None:
        """
        Soft delete — marks session inactive rather than removing from DB.
        Keeps message history intact for audit purposes.
        """
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET is_active = FALSE, updated_at = %s WHERE id = %s",
                    (_now(), session_id),
                )
        logger.info(f"Soft deleted session {session_id[:8]}...")

    def clear_session(self, session_id: str) -> None:
        """
        Delete all messages in a session — 'Clear this chat' action.
        Keeps the session entry itself in the sidebar.
        """
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM messages WHERE session_id = %s",
                    (session_id,),
                )
                cur.execute(
                    "UPDATE sessions SET title = 'New Chat', updated_at = %s WHERE id = %s",
                    (_now(), session_id),
                )
        logger.info(f"Cleared messages for session {session_id[:8]}...")

    # ── Messages ──────────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id:  str,
        role:        str,
        content:     str,
        sources:     list[dict] | None = None,
        confidence:  dict | None = None,
        model_used:  str = "",
    ) -> Message:
        """
        Persist one message. Called after each user question and assistant answer.
        Auto-updates the session's updated_at timestamp.
        """
        now = _now()
        message = Message(
            id=str(uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            sources=sources or [],
            confidence=confidence or {},
            model_used=model_used,
            timestamp=now,
        )

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO messages
                      (id, session_id, role, content, sources, confidence, model_used, timestamp)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    """,
                    (
                        message.id,
                        message.session_id,
                        message.role,
                        message.content,
                        _json(message.sources),
                        _json(message.confidence),
                        message.model_used,
                        message.timestamp,
                    ),
                )
                # Touch session updated_at
                cur.execute(
                    "UPDATE sessions SET updated_at = %s WHERE id = %s",
                    (now, session_id),
                )

        return message

    def get_messages(self, session_id: str) -> list[Message]:
        """
        Return all messages for a session in chronological order.
        Used to restore the full chat history when reopening a tab.
        """
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM messages WHERE session_id = %s ORDER BY timestamp ASC",
                    (session_id,),
                )
                rows = cur.fetchall()
        return [_row_to_message(r) for r in rows]

    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 12,
    ) -> list[dict]:
        """
        Return the last N messages as plain dicts for the LLM prompt.
        Format: [{"role": "user"|"assistant", "content": "..."}]
        This is what gets passed to pipeline.query() as history.
        """
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content FROM messages
                    WHERE session_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (session_id, limit),
                )
                rows = cur.fetchall()

        # Reverse to chronological order (DESC fetch → ASC display)
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ── Row converters ────────────────────────────────────────────────────────────

def _row_to_session(row: dict) -> Session:
    return Session(
        id=row["id"],
        title=row["title"],
        folders=row["folders"],          # psycopg2 auto-parses JSONB → list
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        is_active=row["is_active"],
    )


def _row_to_message(row: dict) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        sources=row["sources"],          # psycopg2 auto-parses JSONB → list
        confidence=row["confidence"],    # psycopg2 auto-parses JSONB → dict
        model_used=row["model_used"],
        timestamp=row["timestamp"],
    )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _now() -> datetime:
    """Current UTC time as a timezone-aware datetime (stored as TIMESTAMPTZ)."""
    return datetime.now(timezone.utc)


def _json(value: list | dict) -> str:
    """Serialise a Python list or dict to a JSON string for the %s::jsonb cast."""
    import json
    return json.dumps(value)
