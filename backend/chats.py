"""Per-user chat conversations. Every query is filtered by username, so users only see their own chats."""
import json
import sqlite3
import time
import uuid

from . import config

_DB = str(config.DATA_DIR / "papers.db")
MAX_CHATS = 50


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY, username TEXT NOT NULL, title TEXT NOT NULL,
        created_at REAL NOT NULL, updated_at REAL NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL, sources TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL)""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(username, updated_at)")
    return c


def create_chat(username: str, title: str = "New chat") -> dict:
    with _conn() as c:
        if c.execute("SELECT COUNT(*) FROM chats WHERE username=?", (username,)).fetchone()[0] >= MAX_CHATS:
            raise ValueError(f"You can have at most {MAX_CHATS} chats. Delete an old one first.")
        cid, now = uuid.uuid4().hex, time.time()
        c.execute("INSERT INTO chats VALUES (?,?,?,?,?)", (cid, username, title[:80], now, now))
    return {"id": cid, "title": title[:80]}


def list_chats(username: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id, title, updated_at FROM chats WHERE username=? ORDER BY updated_at DESC",
                         (username,)).fetchall()
    return [dict(r) for r in rows]


def get_chat(username: str, chat_id: str) -> dict | None:
    """Returns None if the chat does not exist OR belongs to someone else."""
    with _conn() as c:
        chat = c.execute("SELECT id, title FROM chats WHERE id=? AND username=?", (chat_id, username)).fetchone()
        if not chat:
            return None
        msgs = c.execute("SELECT role, content, sources, created_at FROM messages WHERE chat_id=? ORDER BY id",
                         (chat_id,)).fetchall()
    return {"id": chat["id"], "title": chat["title"],
            "messages": [{"role": m["role"], "content": m["content"], "sources": json.loads(m["sources"]),
                          "created_at": m["created_at"]} for m in msgs]}


def add_exchange(username: str, chat_id: str, question: str, answer: str, sources: list) -> bool:
    with _conn() as c:
        if not c.execute("SELECT 1 FROM chats WHERE id=? AND username=?", (chat_id, username)).fetchone():
            return False
        now = time.time()
        c.execute("INSERT INTO messages (chat_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
                  (chat_id, "user", question, "[]", now))
        c.execute("INSERT INTO messages (chat_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
                  (chat_id, "assistant", answer, json.dumps(sources), now))
        c.execute("UPDATE chats SET updated_at=?, title=CASE WHEN title='New chat' THEN ? ELSE title END WHERE id=?",
                  (now, question[:40], chat_id))
    return True


def rename_chat(username: str, chat_id: str, title: str) -> bool:
    with _conn() as c:
        return c.execute("UPDATE chats SET title=? WHERE id=? AND username=?",
                         (title[:80], chat_id, username)).rowcount > 0


def delete_chat(username: str, chat_id: str) -> bool:
    with _conn() as c:  # messages are removed by ON DELETE CASCADE
        return c.execute("DELETE FROM chats WHERE id=? AND username=?", (chat_id, username)).rowcount > 0