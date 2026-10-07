"""Minimal per-profile access control: a random secret issued at profile creation.

The server keeps only a SHA-256 hash. The client keeps the token (localStorage) and sends it in the
X-Token header. This is deliberately simple - enough that one participant cannot read another's
moods by guessing a user id - not a substitute for real accounts.
"""
import hashlib
import hmac
import secrets

from fastapi import HTTPException


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def issue(conn, user_id: int) -> str:
    token = secrets.token_urlsafe(24)
    conn.execute("UPDATE users SET token_hash = %s WHERE user_id = %s", (_hash(token), user_id))
    return token


def verify(conn, user_id: int, token: str | None) -> None:
    row = conn.execute("SELECT token_hash FROM users WHERE user_id = %s", (user_id,)).fetchone()
    if not token or row is None or row[0] is None or not hmac.compare_digest(row[0], _hash(token)):
        raise HTTPException(status_code=403, detail="not your profile")
