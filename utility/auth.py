from sqlalchemy import text
from utility.db import engine


def setup_users_table() -> None:
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id       SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """))
        conn.commit()


def signup(username: str, password: str) -> int:
    with engine.connect() as conn:
        result = conn.execute(
            text("INSERT INTO users (username, password) VALUES (:u, :p) RETURNING id"),
            {"u": username, "p": password},
        )
        conn.commit()
        return result.scalar_one()


def login(username: str, password: str) -> int | None:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT id FROM users WHERE username = :u AND password = :p"),
            {"u": username, "p": password},
        )
        row = result.fetchone()
        return row[0] if row else None
