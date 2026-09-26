import numpy as np
from rank_bm25 import BM25Okapi

from ask import Search, consecutive_runs, expand_neighbors, merge_chunks
from chunking import CHUNK_WORDS, OVERLAP_WORDS, chunk_text


def words(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def fake_search(articles: dict[str, int]) -> Search:
    texts, sources, keys = [], [], []
    for title, n_words in articles.items():
        for position, chunk in enumerate(chunk_text(" ".join(words(title, n_words)), title)):
            texts.append(chunk)
            sources.append({"title": title, "url": title})
            keys.append((title, position))
    return Search(
        texts, np.zeros((len(texts), 1)), sources,
        BM25Okapi([t.split() for t in texts]), keys, {key: i for i, key in enumerate(keys)},
    )


def test_merging_all_chunks_restores_the_article():
    text = " ".join(words("w", 400))
    assert merge_chunks(chunk_text(text, "Roundup"), "Roundup") == "Roundup\n\n" + text


def test_merging_a_slice_keeps_the_overlap_only_once():
    w = words("w", 400)
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks = chunk_text(" ".join(w), "Roundup")
    assert merge_chunks(chunks[1:3], "Roundup") == "Roundup\n\n" + " ".join(w[step : 2 * step + CHUNK_WORDS])


def test_consecutive_runs():
    assert consecutive_runs([5, 1, 2, 3, 7, 8]) == [[1, 2, 3], [5], [7, 8]]


def test_retrieved_chunk_comes_with_its_neighbors():
    # the shipping case: the answer was in the chunk right after the one retrieved
    search = fake_search({"A": 1200})
    hits = expand_neighbors([search.key_index[("A", 1)]], search)
    assert [text for text, _ in hits] == [merge_chunks(search.texts[0:3], "A")]


def test_adjacent_chunks_become_one_passage():
    search = fake_search({"A": 1200})
    hits = expand_neighbors([search.key_index[("A", 1)], search.key_index[("A", 2)]], search)
    assert [text for text, _ in hits] == [merge_chunks(search.texts[0:4], "A")]


def test_distant_chunks_stay_separate():
    search = fake_search({"A": 1200})
    hits = expand_neighbors([search.key_index[("A", 1)], search.key_index[("A", 6)]], search)
    assert len(hits) == 2


def test_window_stops_at_the_end_of_an_article():
    search = fake_search({"A": 1200})
    last = max(position for _, position in search.keys)
    hits = expand_neighbors([search.key_index[("A", last)]], search)
    assert hits[0][0] == merge_chunks(search.texts[last - 1 : last + 1], "A")


def test_passages_keep_retrieval_order():
    search = fake_search({"A": 1200, "B": 400})
    hits = expand_neighbors([search.key_index[("B", 0)], search.key_index[("A", 3)]], search)
    assert [source["title"] for _, source in hits] == ["B", "A"]