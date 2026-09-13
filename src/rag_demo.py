import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3
MIN_SCORE = 0.2  # below this the chunks are noise, see dinner test

SYSTEM = """You answer questions about the game World of Warcraft Forever.
Use ONLY the information in the provided sources. If the sources do not
contain the answer, reply exactly: "I don't have information about that."
Never use prior knowledge."""


def load_chunks(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # paragraph split is enough for a single article, revisit when scraping
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in resp.data])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1))


def retrieve(question: str, chunks: list[str], chunk_vectors: np.ndarray) -> list[str]:
    scores = cosine_similarity(embed([question]), chunk_vectors)[0]
    ranked = np.argsort(scores)[::-1][:TOP_K]
    return [chunks[i] for i in ranked if scores[i] >= MIN_SCORE]


def answer(question: str, sources: list[str]) -> str:
    if not sources:
        return "I don't have information about that."
    source_block = "\n\n---\n\n".join(sources)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"SOURCES:\n{source_block}\n\nQUESTION: {question}"},
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    chunks = load_chunks("data/announcement.txt")
    chunk_vectors = embed(chunks)

    for q in ["When does it release?", "What new race is added?", "What should I eat for dinner?"]:
        print(f"\nQ: {q}")
        print(f"A: {answer(q, retrieve(q, chunks, chunk_vectors))}")