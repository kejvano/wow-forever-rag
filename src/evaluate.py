import argparse
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


def run_case(case: dict, searches: dict) -> dict:
    got, retrieved = ask(case["question"], searches)
    titles = " | ".join(s["title"] for s in retrieved)

    expected_sources = case.get("expect_source", "")
    if isinstance(expected_sources, str):
        expected_sources = [expected_sources] if expected_sources else []
    retrieval_ok = not expected_sources or any(e.lower() in titles.lower() for e in expected_sources)

    return {
        "ok": retrieval_ok and check(case, got),
        "got": got,
        "titles": titles,
        "retrieval_ok": retrieval_ok,
        "expected_sources": expected_sources,
    }


def run(repeat: int = 1) -> int:
    cases = json.loads(Path(QUESTIONS_PATH).read_text(encoding="utf-8"))
    searches = build_searches()

    counts = {"PASS": 0, "FLAKY": 0, "FAIL": 0}
    for case in cases:
        results = [run_case(case, searches) for _ in range(repeat)]
        passes = sum(r["ok"] for r in results)
        label = "PASS" if passes == repeat else "FAIL" if passes == 0 else "FLAKY"
        counts[label] += 1

        # show a failing run when there is one, since that's the one worth reading
        shown = next((r for r in results if not r["ok"]), results[0])
        print(f"{label:5} {passes}/{repeat}  {case['question']}")
        print(f"      got: {shown['got']['answer'][:120].replace(chr(10), ' ')}")
        if shown["got"]["background"]:
            print(f"      background: {shown['got']['background'][:120]}")
        if not shown["retrieval_ok"]:
            print(f"      retrieval missed {shown['expected_sources']}; got: {shown['titles'][:120]}")

    print(f"\n{counts['PASS']} pass, {counts['FLAKY']} flaky, {counts['FAIL']} fail, out of {len(cases)} cases ({repeat} runs each)")
    return 0 if counts["PASS"] == len(cases) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation question set.")
    parser.add_argument("--repeat", type=int, default=1, help="run each case this many times")
    sys.exit(run(parser.parse_args().repeat))