import json
import sys
from pathlib import Path

from ask import ask, build_searches, REFUSAL

QUESTIONS_PATH = "eval/questions.json"


def fragment_found(fragment, text: str) -> bool:
    options = fragment if isinstance(fragment, list) else [fragment]
    return any(str(o).lower() in text.lower() for o in options)


def all_found(fragments, text: str) -> bool:
    if isinstance(fragments, str):
        fragments = [fragments] if fragments else []
    return all(fragment_found(f, text) for f in fragments)


def check(expected: dict, got: dict) -> bool:
    answer_text = got["answer"]
    if expected.get("expect_refusal"):
        answer_ok = answer_text.strip() == REFUSAL
    else:
        answer_ok = all_found(expected["expect"], answer_text)

    if expected.get("expect_no_background"):
        background_ok = not got["background"]
    else:
        background_ok = all_found(expected.get("expect_background", []), got["background"])
    return answer_ok and background_ok


def run() -> int:
    cases = json.loads(Path(QUESTIONS_PATH).read_text(encoding="utf-8"))
    searches = build_searches()

    passed = 0
    for case in cases:
        got, cited = ask(case["question"], searches)
        titles = " | ".join(s["title"] for s in cited)

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