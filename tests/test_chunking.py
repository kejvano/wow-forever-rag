from chunking import CHUNK_WORDS, OVERLAP_WORDS, chunk_text


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_one_chunk_with_title():
    assert chunk_text("hello world", title="Some Article") == ["Some Article\n\nhello world"]


def test_no_title_means_no_prefix():
    assert chunk_text("hello world") == ["hello world"]


def test_long_text_covers_every_word():
    text = words(CHUNK_WORDS * 3 + 17)
    seen = set()
    for chunk in chunk_text(text):
        seen.update(chunk.split())
    assert seen == set(text.split())


def test_chunks_respect_size_limit():
    for chunk in chunk_text(words(CHUNK_WORDS * 3 + 17)):
        assert len(chunk.split()) <= CHUNK_WORDS


def test_consecutive_chunks_overlap():
    chunks = chunk_text(words(CHUNK_WORDS * 3))
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split()[-OVERLAP_WORDS:] == nxt.split()[:OVERLAP_WORDS]


def test_title_is_on_every_chunk():
    chunks = chunk_text(words(CHUNK_WORDS * 2), title="Deep Dive")
    assert len(chunks) > 1
    assert all(chunk.startswith("Deep Dive\n\n") for chunk in chunks)