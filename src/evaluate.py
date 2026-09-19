import json
import sys
from pathlib import Path

from ask import answer, retrieve, build_search, REFUSAL

QUESTIONS_PATH = "eval/questions.json"


def check(expected: dict, got: dict) -> bool:
    answer_text = got["answer"]
    if expected.get("expect_refusal"):
        answer_ok = answer_text.strip() == REFUSAL
    else:
        answer_ok = all(str(s).lower() in answer_text.lower() for s in expected["expect"])
    background_ok = expected.get("expect_background", "").lower() in got["background"].lower()
    return answer_ok and background_ok


def run() -> int:
    cases = json.loads(Path(QUESTIONS_PATH).read_text(encoding="utf-8"))
    texts, vectors, sources, bm25 = build_search()

    passed = 0
    for case in cases:
        hits = retrieve(case["question"], texts, vectors, sources, bm25)
        titles = " | ".join(s["title"] for _, s in hits)
        got = answer(case["question"], hits)

        expected_sources = case.get("expect_source", "")
        if isinstance(expected_sources, str):
            expected_sources = [expected_sources] if expected_sources else []
        retrieval_ok = not expected_sources or any(e.lower() in titles.lower() for e in expected_sources)
        answer_ok = check(case, got)
        ok = retrieval_ok and answer_ok
        passed += ok

        print(f"{'PASS' if ok else 'FAIL'}  {case['question']}")
        print(f"      got: {got['answer'][:120].replace(chr(10), ' ')}")
        if got["background"]:
            print(f"      background: {got['background'][:120]}")
        if not retrieval_ok:
            print(f"      retrieval missed {expected_sources}; got: {titles[:120]}")
    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(run())