import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from chunking import chunk_text
from config import RAW_DIR
from db import connect, article_indexed, insert_article

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 100


def embed(texts: list[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        vectors.extend(item.embedding for item in resp.data)
    return np.array(vectors)


def run() -> None:
    conn = connect()
    indexed = 0
    for meta_path in sorted(Path(RAW_DIR).glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if article_indexed(conn, meta["id"]):
            continue
        text = meta_path.with_suffix(".txt").read_text(encoding="utf-8")
        chunks = chunk_text(text, meta["title"])
        if not chunks:
            continue
        insert_article(conn, meta, chunks, embed(chunks))
        indexed += 1
        print(f"indexed {len(chunks):3d} chunks: {meta['title'] or meta['url']}")
    print(f"done, {indexed} new articles")


if __name__ == "__main__":
    run()