"""Measure how well gap detection matches human-labelled cases.

    python -m eval.run_eval                # offline heuristic baseline
    python -m eval.run_eval --mode llm     # needs ANTHROPIC_API_KEY

Only blocking categories are scored. cases.json is a STARTER set: replace and extend it with
requirements you label yourself (aim for 25+), or the score only measures your own assumptions.
"""
import argparse
import json
import os
from pathlib import Path

BLOCKING = {"actor", "acceptance", "validation", "security"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    args = ap.parse_args()
    os.environ["SPECGUARD_MODE"] = args.mode

    from app.agents import get_backend

    backend = get_backend()
    cases = json.loads((Path(__file__).parent / "cases.json").read_text())
    tp = fp = fn = exact = 0
    for case in cases:
        predicted = {g.id for g in backend.find_gaps(case["requirement"], {}) if g.severity == "blocking"} & BLOCKING
        expected = set(case["missing"])
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        exact += predicted == expected
        if predicted != expected:
            print(f"- {case['requirement'][:60]!r}\n    false alarms: {sorted(predicted - expected) or '-'}  missed: {sorted(expected - predicted) or '-'}")
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    print(f"\nmode={backend.name} cases={len(cases)} precision={precision:.2f} recall={recall:.2f} exact_match={exact}/{len(cases)}")


if __name__ == "__main__":
    main()
