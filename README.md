# wow-forever-rag

![tests](https://github.com/kejvano/wow-forever-rag/actions/workflows/tests.yml/badge.svg)

A question-answering bot for *World of Warcraft: Forever* that keeps itself up to date.

It collects news articles about the game on a schedule, indexes them, and answers questions using only what it has collected, and cites its sources. If the answer isn't in the collected articles, it says so instead of guessing.

![Screenshot](docs/screenshot.png)

```
$ python src/ask.py what is the level cap in beta
The level cap in the WoW: Forever Beta will start at level 20 and then increase
to level 30 at a later time. The cap will remain at 30 for the rest of Beta.
```

```
$ python src/ask.py what should I eat for dinner
I don't have information about that.
```

```
$ python src/ask.py How much does it cost to reset talents when the game is released?
I don't have information about that.

About the original Classic, not confirmed for Forever:
In World of Warcraft Classic, players can reset their talent points at any time,
but the cost for doing so increases with each reset, up to a maximum of 50 gold.
```

## Why

When I started this the announcement for World of Warcraft Forever was very recent and LLMs had not yet been trained on this new information. I saw this as an opportunity to explore something I was interested in: learning how to work with LLMs and experimenting with different ways they could be used. At the same time, I wanted to create something fun that I could share with my friends and that had a practical use.

This project is a retrieval-augmented generation (RAG) pipeline: instead of training a model, the relevant articles are found at question time and handed to the model as context. New articles show up in answers within hours of being published, with no retraining.

## How it works

```
Wowhead RSS feed ─┐
Seed URLs ────────┴─► fetch.py ──► data/raw/forever-news/       (Forever news)
Classic seed URLs ──► fetch.py ──► data/raw/classic-reference/  (Classic reference)
                                           │
                                           ▼
                                      index.py ──► data/index.sqlite   (150-word chunks + embeddings)
                                                           │
                                                           ▼
                         ask.py: query expansion + hybrid retrieval (BM25 + vector) ──► GPT-4o-mini ──► answer
```

**Fetch.** Articles are discovered through RSS feeds and a list of seed URLs for sources without feeds (Blizzard's own news site has none). Article text is extracted with `trafilatura`, preferring full-text feed content over page scraping. Each article is saved once, keyed by a hash of its URL, so the fetcher is safe to re-run at any time. Articles are stored in one subfolder per collection: `forever-news` for coverage of Forever itself, and `classic-reference` for a small, static set of guides to the original game. The folder decides the collection, and news answers are only ever retrieved from `forever-news`.

**Index.** Articles are split into overlapping 150-word chunks, each prefixed with its article title, and embedded with `text-embedding-3-small`. Chunks and vectors are stored in SQLite. The index is derived from the raw layer and can be rebuilt in a minute; changing the chunking never requires re-scraping.

**Ask.** The question is first rewritten by the model into three alternative phrasings, so a question asking about the "max level" can still match a source that says "adventure to level 60". Every phrasing is run through both BM25 keyword search and cosine similarity over the embeddings, and all the rankings are merged with reciprocal rank fusion, with at most three chunks per article so one long article can't fill every slot. Each retrieved chunk is then sent to the model together with its neighboring chunks from the same article, and adjacent chunks are merged into one passage. The model must first copy the sentences from the sources that answer the question, and then write the answer from them. Those sentences are checked against the retrieved text before the answer is shown; if none of them actually appears there, the answer is replaced with a refusal. Only the articles that contain a verified quote are cited under the answer.

**Update.** `update.py` runs fetch and index together and is scheduled every 6 hours through the OS task scheduler, and notifies the running web server to reload the index.

**Background.** If the news sources can't answer a question, the same rewritten queries are run against the Classic reference collection, and a second model call produces background about how the original game handled it. That background goes through the same evidence check as answers do; if no quote verifies, nothing is shown. It is labeled as describing the original Classic, not as confirmed for Forever.


## Setup

Requires Python 3.10 or newer (developed and tested on 3.14) and an OpenAI API key.

```bash
git clone https://github.com/kejvano/wow-forever-rag.git
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
python src/inspect_chunks.py shipping            # shows indexed chunks containing a phrase
```

To keep the index current, schedule `src/update.py`. On Windows this is a Task Scheduler job; on Linux the equivalent cron entry is:

```
0 */6 * * * cd /path/to/wow-forever-rag && .venv/bin/python src/update.py >> logs/update.log 2>&1
```

## Web interface

```bash
uvicorn app:app --app-dir src
```

Open http://127.0.0.1:8000 for a minimal page: type a question, get an answer with the quotes that support it and links to the articles they came from, plus background about the original Classic when the news has no answer. FastAPI's generated API documentation is at http://127.0.0.1:8000/docs, where the endpoints can be tried directly.

| Endpoint | Description |
|---|---|
| `POST /ask` | `{"question": "..."}` → `{"answer", "evidence", "background", "background_evidence", "background_sources", "sources"}` |
| `POST /reload` | Reloads the index from disk; called by `update.py` after each scheduled run so new articles are served without a restart. |

The index is loaded once at startup and held in memory, so nothing is read from the database per request. A question costs one rewrite call, one batched embedding call and one answer call. When the news can't answer, the Classic lookup adds one more embedding call and one more model call. Query expansion roughly doubles latency compared to single-query retrieval, in exchange for far better recall on unusual phrasings.

## Evaluation

`eval/questions.json` holds a set of questions with expected answer fragments and, where known, the article the answer should come from. `python src/evaluate.py` runs them all and reports which cases fail at retrieval and which fail at generation.

This is a regression check, not a benchmark, but it has already earned its keep: with vector-only retrieval and 300-word chunks, the level-cap question failed because the relevant sentence was diluted inside a long chunk about several topics. Smaller chunks fixed it, and adding BM25 made the result stable across chunk sizes.

Refusal cases describe the corpus at a point in time, not permanent truths. Early on, "Will I be able to create characters of different factions on the same account?" had no answer in the sources, and the correct behavior was to refuse. A week later Blizzard published the ruleset details, the scheduled update picked them up, and the bot began answering correctly, which made the old test fail. When a refusal case starts failing after an update, the first thing to check is whether the sources have caught up.

Because the pipeline makes several model calls and the corpus keeps growing, a single run is a sample. `python src/evaluate.py --repeat 3` runs each case three times and labels it PASS, FLAKY or FAIL, and changes are judged by running both versions back to back. This rejected a change that looked like an improvement: giving the query rewriter the game's context produced more readable queries, but turned two reliably passing cases into failures, so it was dropped.

## Tests

`pytest` runs unit tests for the pure parts of the pipeline: chunking, neighbor expansion, the evidence check, tokenization, the source-specific cleanup, and the evaluation's matching rules. They need no API key and run in a few seconds, and GitHub Actions runs them on every push.

They complement the evaluation rather than replace it: the eval measures answer quality end to end but costs API calls and takes minutes, while the unit tests pin down exact behavior cheaply. Several of them are regression tests for real bugs, such as a Blizzard page with invalid structured data and a model quote that merged two sentences. Writing them also found one: curly apostrophes split names like Ula’tek into two tokens, so keyword search missed them.

## Design decisions

- **RAG, not fine-tuning.** The facts change weekly. Fine-tuning would need a retraining cycle per update and still tends to hallucinate specifics; retrieval makes the freshness problem a scheduling problem.
- **Raw layer kept separate from the index.** Scraping is the expensive, fragile step; embedding is cheap. Keeping raw text immutable makes every indexing experiment a one-minute rebuild.
- **SQLite with in-memory vector search.** Under a thousand chunks fit in memory and brute-force cosine similarity runs in microseconds. A vector database would add operational weight for no benefit at this scale; pgvector is the natural next step if the corpus grows by orders of magnitude.
- **Hybrid retrieval.** Embeddings capture meaning but underweight exact terms; the game's vocabulary (ability names, item names, zone names) is exactly what keyword search is good at.
- **Multi-query retrieval.** A question's wording often shares nothing with the source that answers it. Blizzard's announcement says players will "adventure to level 60"; a user asks for the "max level", and neither keyword nor vector search connected them. Rewriting the question into several phrasings before retrieval fixed that class of miss.
- **Verified evidence.** The model must return verbatim quotes supporting its answer, and those quotes are checked against the retrieved chunks in code before the answer is shown. This caught a subtler kind of hallucination than an ungrounded fact: asked whether both factions could exist on one account, the model reasoned from a source saying they cannot group together and answered "yes": fluent, source-flavored, and unsupported. With the check in place it refuses, because no source sentence says it. Quotes are verified sentence by sentence, because the model often merges adjacent sentences into one quote and alters a word in the join; each sentence must be at least six words long, so a trivial fragment can't count as evidence.
- **Refusal over guessing.** If nothing relevant is retrieved, the model isn't called at all. If sources are retrieved but don't contain the answer, the model is instructed to say so.
- **Two collections, two jobs.** Forever news answers questions; a small Classic reference set only supplies background, and each is retrieved separately. Keeping them apart matters: the Warcraft Wiki says the Classic beta had a level cap of 30, which in a shared pool could easily be retrieved for a question about Forever's beta.
- **Small to search, wide to read.** Small chunks make retrieval precise, but a fact can sit just past a chunk boundary. The Collector's Edition shipping cost was in the chunk after the one retrieved, so the model saw "there were some issues with shipping costs" and correctly refused. Retrieved chunks are now expanded with their neighbors before being sent to the model. This costs up to three times the tokens per question, which for gpt-4o-mini is still well under a cent.
- **Quote first, then answer.** The model writes its reply left to right, so when the answer came before the evidence, it answered first and then picked quotes to support what it had already said. That produced answers whose own quotes contradicted them by omission: the evidence said the beta cap starts at 20 and rises to 30, and the answer said 30. With the evidence written first, the answer is built from the quotes rather than justified by them.

## Limitations

- Source coverage is narrow: one feed plus a hand-maintained seed list. Sources without RSS (including Blizzard's own site) are only picked up via seeds or secondhand coverage.
- Site-specific cleanup (`strip_trailing_nav`) is brittle by design and will need maintenance when Wowhead changes its page layout.
- No alerting. If a feed breaks or the API key expires, the only sign is the log.
- The evaluation set is small. It catches regressions on known cases; it doesn't measure overall answer quality.
- Background is verified against Classic sources, but whether a Classic rule carries over to Forever is unknown, and the interface says so. An earlier version generated background from model memory. In testing it claimed Classic allowed both factions on one account, true only outside PvP realms. That failure is why background is now retrieved rather than recalled.
- The evidence check verifies that at least one supporting sentence appears verbatim in the retrieved text; it does not verify every claim in the answer.
- The Warcraft Tavern compendium in the Classic reference set was written shortly before Classic launched in 2019, so a few of its statements are predictions rather than facts.
- Merging chunks relies on the chunk settings the index was built with; after changing them, the index must be rebuilt.

## Possible next steps

- Check in code that every number in an answer appears in a verified quote.
- Local model support via Ollama for fully offline operation.
- Store the embedding model name with the index and refuse to mix models.
- Alerting on failed scheduled runs.
