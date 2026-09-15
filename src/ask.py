import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from db import connect, load_all_chunks
from index import embed

import re
from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
MIN_SCORE = 0.2
CANDIDATES = 20
RRF_K = 60


SYSTEM = """You answer questions about the game World of Warcraft: Forever.
Use ONLY the information in the provided sources. If the sources do not
contain the answer, reply exactly: "I don't have information about that."
Never use prior knowledge. Ignore source content about other games.
Sources are dated. Resolve relative dates like "this Thursday" or
"next week" using the source's publish date, and state the absolute date.
Be complete: if the sources describe a change over time, a condition, or an
exception, include it rather than giving only the final value."""


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "should", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "will", "with", "you",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOPWORDS]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1))


def retrieve(question: str, texts, vectors, sources, bm25: BM25Okapi, debug: bool = False):
    vec_scores = cosine_similarity(embed([question]), vectors)[0]
    vec_ranked = np.argsort(vec_scores)[::-1][:CANDIDATES]

    kw_scores = bm25.get_scores(tokenize(question))
    kw_ranked = np.argsort(kw_scores)[::-1][:CANDIDATES]

    fused: dict[int, float] = {}
    for ranked in (vec_ranked, kw_ranked):
        for rank, i in enumerate(ranked):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)

    top = sorted(fused, key=fused.get, reverse=True)[:TOP_K]
    if debug:
        for i in top:
            print(f"  rrf={fused[i]:.4f} vec={vec_scores[i]:.3f} kw={kw_scores[i]:.2f}  {sources[i]['title'][:60]}")
    return [(texts[i], sources[i]) for i in top if vec_scores[i] >= MIN_SCORE or kw_scores[i] > 0]


def answer(question: str, hits) -> str:
    if not hits:
        return "I don't have information about that."
    source_block = "\n\n---\n\n".join(
        f"[{s['title']}]({s['url']}) — published {s['published'] or 'unknown'}\n{text}" for text, s in hits
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"SOURCES:\n{source_block}\n\nQUESTION: {question}"},
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--debug"]
    debug = "--debug" in sys.argv
    question = " ".join(args) or "When does the game release?"
    conn = connect()
    texts, vectors, sources = load_all_chunks(conn)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    print(f"{len(texts)} chunks loaded")
    print(answer(question, retrieve(question, texts, vectors, sources, bm25, debug)))