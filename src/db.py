import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = "data/index.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id          TEXT PRIMARY KEY,
    collection  TEXT NOT NULL,
    url         TEXT NOT NULL,
    title       TEXT,
    published   TEXT,
    fetched_at  TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    article_id  TEXT NOT NULL REFERENCES articles(id),
    position    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def article_indexed(conn: sqlite3.Connection, article_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM articles WHERE id = ?", (article_id,)).fetchone()
    return row is not None


def insert_article(conn: sqlite3.Connection, meta: dict, chunks: list[str], vectors: np.ndarray) -> None:
    conn.execute(
        "INSERT INTO articles (id, collection, url, title, published, fetched_at) VALUES (?, ?, ?, ?, ?, ?)",
        (meta["id"], meta["collection"], meta["url"], meta["title"], meta["published"], meta["fetched_at"]),
    )
    conn.executemany(
        "INSERT INTO chunks (article_id, position, text, embedding) VALUES (?, ?, ?, ?)",
        [
            (meta["id"], i, text, vec.astype(np.float32).tobytes())
            for i, (text, vec) in enumerate(zip(chunks, vectors))
        ],
    )
    conn.commit()


def load_all_chunks(conn: sqlite3.Connection, collection: str | None = None):
    query = """
        SELECT c.text, c.embedding, a.title, a.url, a.published, a.collection
        FROM chunks c JOIN articles a ON a.id = c.article_id
    """
    params: tuple = ()
    if collection:
        query += " WHERE a.collection = ?"
        params = (collection,)
    rows = conn.execute(query, params).fetchall()
    texts = [r[0] for r in rows]
    vectors = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    sources = [{"title": r[2], "url": r[3], "published": r[4], "collection": r[5]} for r in rows]
    return texts, vectors, sources