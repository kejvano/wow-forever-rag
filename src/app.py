from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ask import build_search, ask

app = FastAPI(title="WoW Forever Q&A")
search = build_search()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_endpoint(q: Question):
    reply, cited = ask(q.question, search)
    return {
      "answer": reply["answer"],
      "background": reply["background"],
      "evidence": reply["evidence"],
      "sources": cited,
    }


@app.post("/reload")
def reload_index():
    global search
    search = build_search()
    return {"chunks": len(search[0])}


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>WoW Forever Q&A</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 3rem auto; padding: 0 1rem; }
  input { width: 100%; font-size: 1.1rem; padding: .6rem; box-sizing: border-box; }
  #answer { margin-top: 1.5rem; white-space: pre-wrap; }
  <div id="evidence"></div>
  #background { margin-top: 1rem; font-style: italic; color: #8a6d3b;
              border-left: 3px solid #d0b070; padding-left: .75rem; }
  #sources { margin-top: 1rem; font-size: .9rem; color: #555; }
  #evidence { margin-top: 1rem; font-size: .9rem; color: #444;
              border-left: 3px solid #bbb; padding-left: .75rem; }
  #evidence p { margin: .4rem 0; }
</style>
</head>
<body>
<h1>WoW: Forever Q&amp;A</h1>
<p>Answers come only from collected news articles. Press Enter to ask.</p>
<input id="q" placeholder="What is the level cap in beta?" autofocus>
<div id="answer"></div>
<div id="background"></div>
<ul id="sources"></ul>
<script>
const q = document.getElementById("q");
q.addEventListener("keydown", async (e) => {
  if (e.key !== "Enter" || !q.value.trim()) return;
  document.getElementById("answer").textContent = "Thinking…";
  document.getElementById("evidence").innerHTML = "";
  document.getElementById("background").textContent = "";
  document.getElementById("sources").innerHTML = "";
  const res = await fetch("/ask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: q.value}),
  });
  const data = await res.json();
  document.getElementById("answer").textContent = data.answer;
  const ev = document.getElementById("evidence");
  for (const quote of data.evidence || []) {
    const p = document.createElement("p");
    p.textContent = "“" + quote + "”";
    ev.appendChild(p);
  }
  document.getElementById("background").textContent =
    data.background ? "Unverified — general Classic knowledge, may be wrong: " + data.background : "";
  for (const s of data.sources) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = s.url; a.textContent = s.title || s.url; a.target = "_blank";
    li.appendChild(a);
    document.getElementById("sources").appendChild(li);
  }
});
</script>
</body>
</html>"""