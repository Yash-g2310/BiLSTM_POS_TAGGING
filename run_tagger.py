#!/usr/bin/env python3
"""Tag a tokenized file using tagger_model.pt.

Required assignment command:
    python run_tagger.py dev.sents dev.out

Blank input lines remain blank output lines. Token spellings and token order are
preserved, and each output token is rendered in `word/TAG` format.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from tagger_lib import MODEL_FILE, load_checkpoint, make_tagged_lines, predict_tags, read_sents


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False in this Python environment")
        device = torch.device(requested)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="one tokenized sentence per line")
    parser.add_argument("output", help="destination for word/TAG predictions")
    parser.add_argument("--model", default=MODEL_FILE, help=f"checkpoint path (default: {MODEL_FILE})")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    model_path, input_path = Path(args.model), Path(args.input)
    if not model_path.exists():
        parser.error(f"Model checkpoint does not exist: {model_path}. Run build_tagger.py first.")
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")

    start = time.perf_counter()
    device = choose_device(args.device)
    model, vocab, cfg, metadata = load_checkpoint(str(model_path), device)
    sentences = read_sents(str(input_path))
    nonempty_indices = [i for i, words in enumerate(sentences) if words]
    nonempty = [sentences[i] for i in nonempty_indices]
    items = [vocab.encode(words) for words in nonempty]
    if items:
        predictions = predict_tags(model, items, vocab, cfg, device, batch_size=args.batch_size)
        tagged = make_tagged_lines(nonempty, predictions, vocab.tags)
    else:
        tagged = []
    by_original_index = dict(zip(nonempty_indices, tagged))
    output_lines = [by_original_index.get(i, "") for i in range(len(sentences))]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    token_count = sum(len(sentence) for sentence in nonempty)
    elapsed = time.perf_counter() - start
    print(f"Device: {device}; sentences: {len(sentences):,}; non-empty: {len(nonempty):,}; tokens: {token_count:,}")
    print(f"Inference elapsed: {elapsed:.2f}s ({token_count / max(elapsed, 1e-9):,.0f} tokens/s including model loading)")
    print(f"Saved predictions: {out_path.resolve()}")


if __name__ == "__main__":
    main()
