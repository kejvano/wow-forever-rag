# wow-forever-rag

A question-answering bot for *World of Warcraft: Forever* that keeps itself up to date.

It collects news articles about the game on a schedule, indexes them, and answers questions using only what it has collected — with sources. If the answer isn't in the collected articles, it says so instead of guessing.

```
$ python src/ask.py what is the level cap in beta
The level cap in the WoW: Forever Beta will start at level 20 and then increase
to level 30 at a later time. The cap will remain at 30 for the rest of Beta.
```

## Why

This project is a retrieval-augmented generation (RAG) pipeline: instead of training a model, the relevant articles are found at question time and handed to the model as context. New articles show up in answers within hours of being published, with no retraining.

I created this project because, at the time, the announcement for World of Warcraft Forever was very recent and LLMs had not yet been trained on this new information. I saw this as an opportunity to explore something I was interested in: learning how to work with LLMs and experimenting with different ways they could be used. At the same time, I wanted to create something fun that I could share with my friends and that had a practical use.

## How it works

```
Wowhead RSS feed ─┐
                  ├─► fetch.py ──► data/raw/*.txt + *.json   (raw articles, source of truth)
Seed URLs ────────┘                        │
                                           ▼
                                      index.py ──► data/index.sqlite   (150-word chunks + embeddings)
                                                           │
                                                           ▼
                             ask.py: hybrid retrieval (BM25 + vector) ──► GPT-4o-mini ──► answer
```

**Fetch.** Articles are discovered through RSS feeds and a list of seed URLs for sources without feeds (Blizzard's own news site has none). Article text is extracted with `trafilatura`, preferring full-text feed content over page scraping. Each article is saved once, keyed by a hash of its URL, so the fetcher is safe to re-run at any time.

**Index.** Articles are split into overlapping 150-word chunks, each prefixed with its article title, and embedded with `text-embedding-3-small`. Chunks and vectors are stored in SQLite. The index is derived from the raw layer and can be rebuilt in a minute; changing the chunking never requires re-scraping.

**Ask.** A question is run through both BM25 keyword search and cosine similarity over the embeddings. The two rankings are merged with reciprocal rank fusion, the top chunks are sent to the model with their source and publish date, and the model is instructed to answer only from those sources.

**Update.** `update.py` runs fetch and index together and is scheduled every 6 hours through the OS task scheduler.

## Setup

Requires Python 3.11+ and an OpenAI API key.

```bash
git clone https://github.com/[your-username]/wow-forever-rag.git
cd wow-forever-rag
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Then populate the index and ask something:

```bash
python src/fetch.py
python src/index.py
python src/ask.py when does the game release
python src/ask.py what is Seal of Fury --debug   # shows retrieval scores
```

To keep the index current, schedule `src/update.py`. On Windows this is a Task Scheduler job; on Linux the equivalent cron entry is:

```
0 */6 * * * cd /path/to/wow-forever-rag && .venv/bin/python src/update.py >> logs/update.log 2>&1
```

## Evaluation

`eval/questions.json` holds a set of questions with expected answer fragments and, where known, the article the answer should come from. `python src/evaluate.py` runs them all and reports which cases fail at retrieval and which fail at generation.

This is a regression check, not a benchmark, but it has already earned its keep: with vector-only retrieval and 300-word chunks, the level-cap question failed because the relevant sentence was diluted inside a long chunk about several topics. Smaller chunks fixed it, and adding BM25 made the result stable across chunk sizes.

## Design decisions

- **RAG, not fine-tuning.** The facts change weekly. Fine-tuning would need a retraining cycle per update and still tends to hallucinate specifics; retrieval makes the freshness problem a scheduling problem.
- **Raw layer kept separate from the index.** Scraping is the expensive, fragile step; embedding is cheap. Keeping raw text immutable makes every indexing experiment a one-minute rebuild.
- **SQLite with in-memory vector search.** A few hundred chunks fit in memory and brute-force cosine similarity runs in microseconds. A vector database would add operational weight for no benefit at this scale; pgvector is the natural next step if the corpus grows by orders of magnitude.
- **Hybrid retrieval.** Embeddings capture meaning but underweight exact terms; the game's vocabulary (ability names, item names, zone names) is exactly what keyword search is good at.
- **Refusal over guessing.** If nothing relevant is retrieved, the model isn't called at all. If sources are retrieved but don't contain the answer, the model is instructed to say so.

## Limitations

- Source coverage is narrow: one feed plus a hand-maintained seed list. Sources without RSS (including Blizzard's own site) are only picked up via seeds or secondhand coverage.
- Site-specific cleanup (`strip_trailing_nav`) is brittle by design and will need maintenance when Wowhead changes its page layout.
- No alerting. If a feed breaks or the API key expires, the only sign is the log.
- The evaluation set is small. It catches regressions on known cases; it doesn't measure overall answer quality.
- Overlapping chunks from the same article can both be retrieved, occasionally biasing the answer toward whichever phrasing appears twice.

## Possible next steps

- Web interface (FastAPI) so the bot can be used without a terminal.
- Merge adjacent chunks from the same article before sending them to the model.
- Local model support via Ollama for fully offline operation.
- Store the embedding model name with the index and refuse to mix models.
- Alerting on failed scheduled runs.