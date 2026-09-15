import json
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

from ask import answer, retrieve, tokenize, REFUSAL
from db import connect, load_all_chunks

QUESTIONS_PATH = "eval/questions.json"


def check(expected: dict, got: str) -> bool:
    if expected.get("expect_refusal"):
        return got.strip() == REFUSAL
    return all(s.lower() in got.lower() for s in expected["expect"])


def run() -> int:
    cases = json.loads(Path(QUESTIONS_PATH).read_text(encoding="utf-8"))
    conn = connect()
    texts, vectors, sources = load_all_chunks(conn)
    bm25 = BM25Okapi([tokenize(t) for t in texts])

    passed = 0
    for case in cases:
        got = answer(case["question"], retrieve(case["question"], texts, vectors, sources, bm25))
        ok = check(case, got)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['question']}")
        if not ok:
            print(f"      got: {got[:200]}")
    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(run())