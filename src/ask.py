import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from db import connect, load_all_chunks
from index import embed

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
MIN_SCORE = 0.2

SYSTEM = """You answer questions about the game World of Warcraft: Forever.
Use ONLY the information in the provided sources. If the sources do not
contain the answer, reply exactly: "I don't have information about that."
Never use prior knowledge. Ignore source content about other games.
Sources are dated. Resolve relative dates like "this Thursday" or
"next week" using the source's publish date, and state the absolute date."""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1))


def retrieve(question: str, texts, vectors, sources, debug: bool = False):
    scores = cosine_similarity(embed([question]), vectors)[0]
    ranked = np.argsort(scores)[::-1][:TOP_K]
    if debug:
        for i in ranked:
            print(f"  {scores[i]:.3f}  {sources[i]['title'][:70]}")
    return [(texts[i], sources[i]) for i in ranked if scores[i] >= MIN_SCORE]


def answer(question: str, hits) -> str:
    if not hits:
        return "I don't have information about that."
    source_block = "\n\n---\n\n".join(
        f"[{s['title']}]({s['url']}) — published {s['published'] or 'unknown'}\n{text}" for text, s in hits
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
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
    print(f"{len(texts)} chunks loaded")
    print(answer(question, retrieve(question, texts, vectors, sources, debug)))