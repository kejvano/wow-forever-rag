import os
import sys
import json
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config import NEWS
from db import connect, load_all_chunks
from index import embed

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CHAT_MODEL = "gpt-4o-mini"
TOP_K = 10
MIN_SCORE = 0.2
CANDIDATES = 30
RRF_K = 60
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_QUOTE_WORDS = 6
REFUSAL = "I don't have information about that."


SYSTEM = f"""You answer questions about the game World of Warcraft: Forever, a
new version of the game based on the original 2004 "Classic" World of Warcraft.
Distinguish beta from the released game. Statements about the beta
(level caps, known issues, dates) do not describe launch unless the source says so.

Respond with a JSON object with exactly three keys:

"answer": what the provided sources say about the question. Rules:
- Use ONLY what the sources directly state. Do not infer or extrapolate: a
  source describing one thing (for example, that factions cannot group
  together) does not tell you about a related thing (for example, whether both
  factions can exist on one account).
- A question about whether something is possible or allowed requires a source
  that directly states it. A source saying two things cannot interact does not
  establish that both can exist.
- When the sources give a value that changes over time, state the starting
  value and the later value, not just the final one. Every number that appears
  in your evidence must appear in your answer.
- Resolve relative dates using the source's publish date.
- Include conditions and exceptions.
- Ignore source content about other games.
- If no source directly addresses the question, this value must be exactly
  "{REFUSAL}".

"evidence": a list of one to three sentences copied exactly, word for word,
from the sources. Together they must support every part of the answer: if the
answer mentions a starting value and a later value, quote both sentences. If
the answer is the refusal, this is an empty list.

"background": this key is NOT subject to the source-only rules above; it is
explicitly labeled to the user as general knowledge rather than reporting.
Fill it whenever the answer is the refusal or is incomplete, and the original
Classic World of Warcraft (2019) has a well-known relevant answer — for
example, that Classic's level cap was 60, or how Classic handled factions on
an account. Write it as what Classic did, never as fact about Forever, and
never contradict the sources. One or two sentences. Empty only when Classic
offers nothing relevant. Prefer to omit a detail rather than guess at it;
where a rule varied (for example between realm types), say so or leave it out.

Output only the JSON object, no markdown fences."""

REWRITE_SYSTEM = """Rewrite the user's question as 3 short alternative search
queries that use different wording and likely vocabulary from game news
articles (official terms, synonyms, concrete numbers where implied). Return a
JSON object {"queries": [...]}. Do not answer the question."""


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "should", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "will", "with", "you",
}


PUNCT = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOPWORDS]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1))


def retrieve(question: str, texts, vectors, sources, bm25: BM25Okapi, debug: bool = False):
    queries = expand_query(question)
    if debug:
        print(f"  queries: {queries}")

    fused: dict[int, float] = {}
    best_vec = np.zeros(len(texts))
    best_kw = np.zeros(len(texts))
    for q in queries:
        vec_scores = cosine_similarity(embed([q]), vectors)[0]
        kw_scores = bm25.get_scores(tokenize(q))
        best_vec = np.maximum(best_vec, vec_scores)
        best_kw = np.maximum(best_kw, kw_scores)
        for ranked in (np.argsort(vec_scores)[::-1][:CANDIDATES], np.argsort(kw_scores)[::-1][:CANDIDATES]):
            for rank, i in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)

    top = sorted(fused, key=lambda i: fused[i], reverse=True)[:TOP_K]
    if debug:
        for i in top:
            print(f"  rrf={fused[i]:.4f} vec={best_vec[i]:.3f} kw={best_kw[i]:.2f}  {sources[i]['title'][:60]}")
    return [(texts[i], sources[i]) for i in top if best_vec[i] >= MIN_SCORE or best_kw[i] > 0]


def expand_query(question: str) -> list[str]:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        seed=42,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    content = resp.choices[0].message.content
    try:
        extra = json.loads(content or "{}").get("queries", [])
    except json.JSONDecodeError:
        extra = []
    return [question] + [q for q in extra if isinstance(q, str) and q.strip()]


def answer(question: str, hits, debug: bool = False) -> dict:
    if not hits:
        return {"answer": REFUSAL, "background": "", "evidence": []}
    source_block = "\n\n---\n\n".join(
        f"[{s['title']}]({s['url']}) — published {s['published'] or 'unknown'}\n{text}"
        for text, s in hits
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        seed=42,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"SOURCES:\n{source_block}\n\nQUESTION: {question}"},
        ],
    )
    content = resp.choices[0].message.content
    if not content:
        return {"answer": REFUSAL, "background": "", "evidence": []}
    data = json.loads(content)
    reply = data.get("answer", REFUSAL)
    evidence = data.get("evidence", [])
    supported = evidence_supported(evidence, hits)
    if debug:
        for q in evidence:
            print(f"  evidence {'ok' if normalize(q) in normalize(' '.join(t for t, _ in hits)) else '??'}: {q[:100]}")
        print(f"  evidence verdict: {'supported' if supported else 'rejected'}")
    if reply != REFUSAL and not supported:
        reply = REFUSAL
    return {"answer": reply, "background": data.get("background", ""), "evidence": evidence}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(PUNCT).lower()).strip()


def ngrams(words: list[str], n: int = 3) -> set[tuple]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def quote_supported(quote: str, corpus: str, corpus_grams: set, min_overlap: float) -> bool:
    q = normalize(quote)
    words = q.split()
    if len(words) < MIN_QUOTE_WORDS:
        return False
    if q in corpus:
        return True
    grams = ngrams(words)
    return bool(grams) and len(grams & corpus_grams) / len(grams) >= min_overlap


def evidence_supported(evidence: list, hits, min_overlap: float = 0.95) -> bool:
    if not evidence:
        return False
    corpus = normalize(" ".join(text for text, _ in hits))
    corpus_grams = ngrams(corpus.split())
    for quote in evidence:
        for sentence in SENTENCE_SPLIT.split(quote):
            if quote_supported(sentence, corpus, corpus_grams, min_overlap):
                return True
    return False


def build_search():
    conn = connect()
    texts, vectors, sources = load_all_chunks(conn, NEWS)
    if not texts:
        raise RuntimeError(f"No chunks in collection '{NEWS}'. Run fetch.py and index.py first.")
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return texts, vectors, sources, bm25


def ask(question: str, search, debug: bool = False) -> tuple[dict, list[dict]]:
    texts, vectors, sources, bm25 = search
    hits = retrieve(question, texts, vectors, sources, bm25, debug)
    reply = answer(question, hits, debug)
    cited = []
    for _, s in hits:
        if s not in cited:
            cited.append(s)
    return reply, cited


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--debug"]
    debug = "--debug" in sys.argv
    question = " ".join(args) or "When does the game release?"
    search = build_search()
    print(f"{len(search[0])} chunks loaded")
    reply, cited = ask(question, search, debug)
    print(reply["answer"])
    if reply["background"]:
        print(f"\nUnverified — general Classic knowledge, not from sources and may be wrong:\n{reply['background']}")
    for s in cited:
        print(f"  - {s['title']} ({s['url']})")