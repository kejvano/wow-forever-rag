CHUNK_WORDS = 300
OVERLAP_WORDS = 30


def chunk_text(text: str, title: str = "") -> list[str]:
    words = text.split()
    if not words:
        return []
    prefix = f"{title}\n\n" if title else ""
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_WORDS
        chunks.append(prefix + " ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - OVERLAP_WORDS
    return chunks