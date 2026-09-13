import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("data/announcement.txt", encoding="utf-8") as f:
    document = f.read()

# paragraph split is fine for a single article
chunks = [p.strip() for p in document.split("\n\n") if p.strip()]
print(f"{len(chunks)} chunks")

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([item.embedding for item in resp.data])

chunk_vectors = embed(chunks)
print("vector shape:", chunk_vectors.shape)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1))

question = "When does the game release?"
q_vector = embed([question])

scores = cosine_similarity(q_vector, chunk_vectors)[0]
top = np.argsort(scores)[::-1][:3]

for i in top:
    print(f"\n--- score {scores[i]:.3f} ---\n{chunks[i]}")

lowest = np.argmin(scores)
print(f"\n--- lowest score {scores[lowest]:.3f} ---\n{chunks[lowest]}")