import os
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import update
from ask import ask, build_searches
from config import NEWS
from db import DB_PATH
from ratelimit import DailyCap, RateLimiter

# 0 turns the built-in updater off, which is what you want when developing locally
UPDATE_INTERVAL_HOURS = float(os.environ.get("UPDATE_INTERVAL_HOURS", "0"))
FIRST_UPDATE_DELAY_SECONDS = 60
MAX_QUESTION_CHARS = 300

per_visitor = RateLimiter([
    (60, int(os.environ.get("VISITOR_PER_MINUTE", "5"))),
    (86400, int(os.environ.get("VISITOR_PER_DAY", "30"))),
])
daily_cap = DailyCap(int(os.environ.get("DAILY_QUESTION_CAP", "300")))


def db_mtime() -> int | None:
    try:
        return Path(DB_PATH).stat().st_mtime_ns
    except FileNotFoundError:
        return None


class LiveIndex:
    # rebuilds the in-memory search whenever index.sqlite changes on disk,
    # whether the scheduled update or a manual update.py run wrote it
    def __init__(self):
        self.searches = None
        self.loaded_mtime = None
        self.updating = False
        self.lock = threading.Lock()

    def current(self):
        mtime = db_mtime()
        if mtime is None or mtime == self.loaded_mtime or self.updating:
            return self.searches
        with self.lock:
            if mtime != self.loaded_mtime:
                try:
                    self.searches = build_searches()
                    self.loaded_mtime = mtime
                    print(f"index loaded: {len(self.searches[NEWS].texts)} news chunks", flush=True)
                except (RuntimeError, sqlite3.OperationalError) as e:
                    print(f"index not loaded: {e}", flush=True)
        return self.searches


live_index = LiveIndex()


def update_loop(stop: threading.Event) -> None:
    delay = FIRST_UPDATE_DELAY_SECONDS
    while not stop.wait(delay):
        live_index.updating = True
        try:
            update.run()
        except Exception as e:
            # a failed run must not end the loop; the next one may well succeed
            print(f"scheduled update failed: {e!r}", flush=True)
        finally:
            live_index.updating = False
        live_index.current()
        delay = UPDATE_INTERVAL_HOURS * 3600


@asynccontextmanager
async def lifespan(app: FastAPI):
    live_index.current()
    stop = threading.Event()
    # runs in this process, so the app must run as a single worker
    if UPDATE_INTERVAL_HOURS > 0:
        threading.Thread(target=update_loop, args=(stop,), daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="WoW Forever Q&A", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


def client_ip(request: Request) -> str:
    # Render puts the real client address first in X-Forwarded-For
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


@app.post("/ask")
def ask_endpoint(q: Question, request: Request):
    if not per_visitor.allow(client_ip(request)):
        raise HTTPException(429, "You're asking faster than the bot allows. Wait a minute and try again.")
    searches = live_index.current()
    if searches is None:
        raise HTTPException(503, "The bot is still setting up. Try again in a few minutes.")
    if not daily_cap.allow():
        raise HTTPException(429, "The bot has answered as many questions as it can today. Come back tomorrow.")
    try:
        reply, _ = ask(q.question.strip(), searches)
    except openai.OpenAIError as e:
        print(f"model call failed: {e!r}", flush=True)
        raise HTTPException(503, "The answering service is unavailable right now. Try again later.")
    return reply


@app.get("/health")
def health():
    # the platform polls this, so it also picks up an index that appeared on disk
    searches = live_index.current()
    return {"ready": searches is not None, "news_chunks": len(searches[NEWS].texts) if searches else 0}


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WoW Forever Q&amp;A</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 3rem auto; padding: 0 1rem; }
  input { width: 100%; font-size: 1.1rem; padding: .6rem; box-sizing: border-box; }
  #answer { margin-top: 1.5rem; white-space: pre-wrap; }
  #evidence { margin-top: 1rem; font-size: .9rem; color: #444;
              border-left: 3px solid #bbb; padding-left: .75rem; }
  #evidence p { margin: .4rem 0; }
  #sources { margin-top: 1rem; font-size: .9rem; color: #555; }
  #background { margin-top: 1.5rem; font-style: italic; color: #8a6d3b;
                border-left: 3px solid #d0b070; padding-left: .75rem; }
  #background p { margin: .4rem 0; }
  footer { margin-top: 3rem; font-size: .8rem; color: #777; }
</style>
</head>
<body>
<h1>WoW: Forever Q&amp;A</h1>
<p>Answers come only from collected news articles. When the news is silent, background about the original Classic may be shown, taken from Classic guides. Press Enter to ask.</p>
<input id="q" placeholder="What is the level cap in beta?" maxlength="300" autofocus>
<div id="answer"></div>
<div id="evidence"></div>
<ul id="sources"></ul>
<div id="background"></div>
<footer>A hobby project, not affiliated with Blizzard. Answers can be wrong, so check the linked sources. Questions are rate limited.</footer>
<script>
const q = document.getElementById("q");
const answerEl = document.getElementById("answer");
const evidenceEl = document.getElementById("evidence");
const sourcesEl = document.getElementById("sources");
const backgroundEl = document.getElementById("background");
let busy = false;

function link(source) {
  const a = document.createElement("a");
  a.href = source.url;
  a.textContent = source.title || source.url;
  a.target = "_blank";
  a.rel = "noopener";
  return a;
}

q.addEventListener("keydown", async (e) => {
  if (e.key !== "Enter" || !q.value.trim() || busy) return;
  busy = true;
  answerEl.textContent = "Thinking…";
  evidenceEl.innerHTML = "";
  sourcesEl.innerHTML = "";
  backgroundEl.innerHTML = "";

  let res, data;
  try {
    res = await fetch("/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: q.value}),
    });
    data = await res.json();
  } catch (err) {
    answerEl.textContent = "Could not reach the server. Try again in a moment.";
    busy = false;
    return;
  }
  busy = false;

  if (!res.ok) {
    answerEl.textContent = typeof data.detail === "string"
      ? data.detail
      : "Questions can be at most 300 characters.";
    return;
  }

  answerEl.textContent = data.answer;
  for (const quote of data.evidence || []) {
    const p = document.createElement("p");
    p.textContent = "“" + quote + "”";
    evidenceEl.appendChild(p);
  }
  for (const s of data.sources || []) {
    const li = document.createElement("li");
    li.appendChild(link(s));
    sourcesEl.appendChild(li);
  }
  if (data.background) {
    const p = document.createElement("p");
    p.textContent = "About the original Classic, not confirmed for Forever: " + data.background;
    backgroundEl.appendChild(p);
    for (const s of data.background_sources || []) {
      const div = document.createElement("div");
      div.appendChild(link(s));
      backgroundEl.appendChild(div);
    }
  }
});
</script>
</body>
</html>"""
