#!/usr/bin/env python3
"""Evaluate a tagged prediction file against a gold `word/TAG` file safely."""
from __future__ import annotations

import argparse
from pathlib import Path

from tagger_lib import read_tagged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predicted", help="model output file, word/TAG format")
    parser.add_argument("gold", help="gold file, word/TAG format")
    args = parser.parse_args()
    pred_path, gold_path = Path(args.predicted), Path(args.gold)
    predictions = read_tagged(str(pred_path))
    gold = read_tagged(str(gold_path))
    if len(predictions) != len(gold):
        raise SystemExit(f"Sentence-count mismatch: predictions={len(predictions)}, gold={len(gold)}")
    token_count = correct = exact = 0
    for i, ((pred_words, pred_tags), (gold_words, gold_tags)) in enumerate(zip(predictions, gold), start=1):
        if pred_words != gold_words:
            mismatch = next((j for j, (a, b) in enumerate(zip(pred_words, gold_words), start=1) if a != b), None)
            raise SystemExit(f"Tokenization mismatch in sentence {i} (first differing token: {mismatch}); prediction/gold are not aligned")
        if len(pred_tags) != len(gold_tags):
            raise SystemExit(f"Token/tag count mismatch in sentence {i}")
        token_count += len(gold_tags)
        n_ok = sum(p == g for p, g in zip(pred_tags, gold_tags))
        correct += n_ok
        exact += int(n_ok == len(gold_tags))
    print(f"Token accuracy: {correct / token_count:.6f} ({correct:,}/{token_count:,})")
    print(f"Exact-sentence accuracy: {exact / len(gold):.6f} ({exact:,}/{len(gold):,})")


if __name__ == "__main__":
    main()
