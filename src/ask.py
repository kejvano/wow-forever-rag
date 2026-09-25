import json
import os
import re
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config import NEWS, CLASSIC
from db import connect, load_all_chunks
from index import embed

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CHAT_MODEL = "gpt-4o-mini"
TOP_K = 10
MIN_SCORE = 0.2
CANDIDATES = 30
RRF_K = 60
MIN_QUOTE_WORDS = 6
REFUSAL = "I don't have information about that."


SYSTEM = f"""You answer questions about the game World of Warcraft: Forever, a
new version of the game based on the original 2004 "Classic" World of Warcraft.
Distinguish beta from the released game. Statements about the beta
(level caps, known issues, dates) do not describe launch unless the source says so.

Respond with a JSON object with exactly two keys:

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

Output only the JSON object, no markdown fences."""

BACKGROUND_SYSTEM = """A question about World of Warcraft: Forever could not be
answered from Forever news. You are given sources about the original 2004
World of Warcraft ("Classic") instead, and provide background from them.

Respond with a JSON object with exactly two keys:

"background": what the provided Classic sources say that helps with the
question, in one or two sentences. Rules:
- Use ONLY what the sources directly state. Do not infer or extrapolate.
- Write it as what Classic did, never as fact about Forever.
- Include conditions and exceptions, such as rules that differed between
  realm types.
- If the sources do not address the question, or the question is not about
  World of Warcraft, this value must be an empty string.

"evidence": a list of one to three sentences copied exactly, word for word,
from the sources, supporting the background. If the background is empty, this
is an empty list.

Output only the JSON object, no markdown fences."""

REWRITE_SYSTEM = """Rewrite the user's question about World of Warcraft (Forever or the original
Classic) as 3 short alternative search queries that use different wording and likely vocabulary
from game news articles (official terms, synonyms, concrete numbers where implied). Return a
JSON object {"queries": [...]}. Do not answer the question."""


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "should", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "will", "with", "you",
}

PUNCT = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.translate(PUNCT).lower()) if t not in STOPWORDS]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1))


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
        for sentence in SENTENCE_SPLIT.split(str(quote)):
            if quote_supported(sentence, corpus, corpus_grams, min_overlap):
                return True
    return False


def complete_json(system: str, user: str) -> dict:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        seed=42,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    if not content:
        return {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def expand_query(question: str) -> list[str]:
    extra = complete_json(REWRITE_SYSTEM, question).get("queries", [])
    if not isinstance(extra, list):
        extra = []
    return [question] + [q for q in extra if isinstance(q, str) and q.strip()]


def retrieve(queries: list[str], search, debug: bool = False):
    texts, vectors, sources, bm25 = search
    query_vectors = embed(queries)

    fused: dict[int, float] = {}
    best_vec = np.zeros(len(texts))
    best_kw = np.zeros(len(texts))
    for q, q_vec in zip(queries, query_vectors):
        vec_scores = cosine_similarity(q_vec[None, :], vectors)[0]
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


def format_sources(hits) -> str:
    return "\n\n---\n\n".join(
        f"[{s['title']}]({s['url']}) — published {s['published'] or 'unknown'}\n{text}"
        for text, s in hits
    )


def unique_sources(hits) -> list[dict]:
    seen = []
    for _, s in hits:
        if s not in seen:
            seen.append(s)
    return seen


def print_evidence(label: str, evidence: list, supported: bool, hits) -> None:
    corpus = normalize(" ".join(t for t, _ in hits))
    for q in evidence:
        mark = "ok" if normalize(str(q)) in corpus else "??"
        print(f"  {label} evidence {mark}: {str(q)[:100]}")
    print(f"  {label} verdict: {'supported' if supported else 'rejected'}")


def answer(question: str, hits, debug: bool = False) -> dict:
    if not hits:
        return {"answer": REFUSAL, "evidence": []}
    data = complete_json(SYSTEM, f"SOURCES:\n{format_sources(hits)}\n\nQUESTION: {question}")
    reply = data.get("answer", REFUSAL)
    evidence = data.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    supported = evidence_supported(evidence, hits)
    if debug:
        print_evidence("answer", evidence, supported, hits)
    if reply == REFUSAL or not supported:
        return {"answer": REFUSAL, "evidence": []}
    return {"answer": reply, "evidence": evidence}


def background(question: str, hits, debug: bool = False) -> dict:
    empty = {"background": "", "background_evidence": [], "background_sources": []}
    if not hits:
        return empty
    data = complete_json(BACKGROUND_SYSTEM, f"SOURCES:\n{format_sources(hits)}\n\nQUESTION: {question}")
    text = data.get("background", "")
    evidence = data.get("evidence", [])
    if not isinstance(text, str) or not isinstance(evidence, list):
        return empty
    supported = evidence_supported(evidence, hits)
    if debug:
        print_evidence("background", evidence, supported, hits)
    if not text.strip() or not supported:
        return empty
    return {"background": text, "background_evidence": evidence, "background_sources": unique_sources(hits)}


def build_search(collection: str):
    conn = connect()
    texts, vectors, sources = load_all_chunks(conn, collection)
    if not texts:
        raise RuntimeError(f"No chunks in collection '{collection}'. Run fetch.py and index.py first.")
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return texts, vectors, sources, bm25


def build_searches() -> dict:
    return {NEWS: build_search(NEWS), CLASSIC: build_search(CLASSIC)}


def ask(question: str, searches: dict, debug: bool = False) -> tuple[dict, list[dict]]:
    queries = expand_query(question)
    if debug:
        print(f"  queries: {queries}")

    news_hits = retrieve(queries, searches[NEWS], debug)
    reply = answer(question, news_hits, debug)

    # background only when the news can't answer; otherwise it's mostly noise
    if reply["answer"] == REFUSAL:
        if debug:
            print("  -- classic reference --")
        reply.update(background(question, retrieve(queries, searches[CLASSIC], debug), debug))
    else:
        reply.update({"background": "", "background_evidence": [], "background_sources": []})

    return reply, unique_sources(news_hits)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--debug"]
    debug = "--debug" in sys.argv
    question = " ".join(args) or "When does the game release?"
    searches = build_searches()
    print(f"{len(searches[NEWS][0])} news chunks, {len(searches[CLASSIC][0])} classic chunks loaded")
    reply, cited = ask(question, searches, debug)
    print(reply["answer"])
    for s in cited:
        print(f"  - {s['title']} ({s['url']})")
    if reply["background"]:
        print(f"\nAbout the original Classic, not confirmed for Forever:\n{reply['background']}")
        for s in reply["background_sources"]:
            print(f"  - {s['title']} ({s['url']})")