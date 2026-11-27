"""Users, password hashing, JWT login tokens and role checks (roles: admin, user)."""
import hashlib
import hmac
import os
import re
import sqlite3
import time

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import config

_bearer = HTTPBearer(auto_error=False)
_DB = str(config.DATA_DIR / "papers.db")
_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_fails: dict[str, list[float]] = {}  # failed logins per ip+username
MAX_FAILS, WINDOW = 5, 300           # 5 failures per 5 minutes


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('admin', 'user')), created_at REAL NOT NULL)""")
    return c


def init() -> None:
    """Called at startup: refuse to run with a weak or missing signing secret."""
    if len(config.JWT_SECRET) < 32:
        raise RuntimeError('JWT_SECRET is missing or shorter than 32 characters. Generate one with: '
                           'python -c "import secrets; print(secrets.token_urlsafe(48))"')
    _conn().close()


def _hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _verify(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
        return hmac.compare_digest(_hash(password, bytes.fromhex(salt)).split("$")[2], digest)
    except Exception:
        return False


_DUMMY = _hash("dummy-password")  # checked when the user does not exist, so timing does not reveal it


def create_user(username: str, password: str, role: str = "user", update: bool = False) -> None:
    username = username.strip().lower()
    if not _USERNAME.match(username):
        raise ValueError("Username must be 3-32 characters: letters, digits, _ . -")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    if role not in ("admin", "user"):
        raise ValueError("Role must be admin or user")
    with _conn() as c:
        if c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone() and not update:
            raise ValueError(f"User '{username}' already exists (use --update to change it)")
        c.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?)", (username, _hash(password), role, time.time()))


def get_user(username: str):
    with _conn() as c:
        return c.execute("SELECT * FROM users WHERE username=?", (username.strip().lower(),)).fetchone()


def authenticate(username: str, password: str, ip: str = "") -> dict | None:
    key = f"{ip}|{username.strip().lower()}"
    now = time.time()
    if len(_fails) > 10000:
        _fails.clear()
    recent = [t for t in _fails.get(key, []) if now - t < WINDOW]
    if len(recent) >= MAX_FAILS:
        raise HTTPException(429, "Too many failed attempts. Try again in a few minutes.")
    row = get_user(username)
    ok = _verify(password, row["password_hash"] if row else _DUMMY)
    if not (row and ok):
        _fails[key] = recent + [now]
        return None
    _fails.pop(key, None)
    return {"username": row["username"], "role": row["role"]}


def make_token(username: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": username, "iat": now, "exp": now + config.JWT_EXPIRE_HOURS * 3600},
                      config.JWT_SECRET, algorithm="HS256")


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    if not creds:
        raise HTTPException(401, "Not logged in", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(creds.credentials, config.JWT_SECRET, algorithms=["HS256"],
                             options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        raise HTTPException(401, "Session expired or invalid. Please log in again.")
    row = get_user(str(payload["sub"]))  # role is read from the database on every request
    if not row:
        raise HTTPException(401, "Unknown user")
    return {"username": row["username"], "role": row["role"]}


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "Admin only")
    return user
