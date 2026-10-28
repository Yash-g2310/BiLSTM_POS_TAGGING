#!/usr/bin/env python3
"""PA2-compatible accuracy evaluator with alignment checks.

Usage: python MeasurePOSTaggerAccuracy.py dev.out data/dev.tags
Reports token accuracy in a familiar `Accuracy= ...` format.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from tagger_lib import read_tagged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predicted", help="prediction file in word/TAG format")
    parser.add_argument("gold", help="reference file in word/TAG format")
    args = parser.parse_args()
    predicted = read_tagged(str(Path(args.predicted)))
    gold = read_tagged(str(Path(args.gold)))
    if len(predicted) != len(gold):
        raise SystemExit(f"Sentence count mismatch: predicted={len(predicted)} gold={len(gold)}")
    total = correct = 0
    for i, ((pw, pt), (gw, gt)) in enumerate(zip(predicted, gold), start=1):
        if pw != gw:
            raise SystemExit(f"Tokenization mismatch on sentence {i}; refusing to compare misaligned tags")
        if len(pt) != len(gt):
            raise SystemExit(f"Token/tag count mismatch on sentence {i}")
        total += len(gt)
        correct += sum(p == g for p, g in zip(pt, gt))
    if total == 0:
        raise SystemExit("No tokens to evaluate")
    accuracy = correct / total
    print(f"Accuracy= {accuracy:.8f} ({correct}/{total})")


if __name__ == "__main__":
    main()
