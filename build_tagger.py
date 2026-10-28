#!/usr/bin/env python3
"""Train a PA2-compliant POS tagger.

Required assignment command:
    python build_tagger.py train.tags

The saved model (default: tagger_model.pt) is written in the current working
 directory. By default, early stopping selects the best epoch using a sentence-
level holdout from the supplied training corpus. No pretrained vectors are used.
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch

from tagger_lib import (
    DEFAULT_CFG,
    MODEL_FILE,
    Vocab,
    read_tagged,
    save_checkpoint,
    split_train_val,
    train_model,
    validate_config,
)


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


def load_config(config_path: str | None) -> dict:
    cfg = dict(DEFAULT_CFG)
    if not config_path:
        default_selected = Path("best_config.json")
        if default_selected.exists():
            config_path = str(default_selected)
        else:
            return cfg
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "cfg" in data and isinstance(data["cfg"], dict):
        data = data["cfg"]
    unknown = set(data) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown config key(s) in {path}: {sorted(unknown)}")
    cfg.update(data)
    validate_config(cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train", nargs="?", default="data/train.tags", help="word/tag training file (default: data/train.tags)")
    parser.add_argument("--config", default=None, help="JSON model config; defaults to the built-in configuration")
    parser.add_argument("--model-out", default=MODEL_FILE, help=f"checkpoint path (default: {MODEL_FILE}, in CWD)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--time-budget", type=float, default=840.0, help="soft training time guard in seconds; default 14 minutes")
    parser.add_argument("--max-sentences", type=int, default=None, help="limit to the first N real training sentences for smoke testing only")
    args = parser.parse_args()

    started = time.perf_counter()
    data_path = Path(args.train)
    if not data_path.exists():
        parser.error(f"Training file does not exist: {data_path}")
    all_sentences = read_tagged(str(data_path))
    if args.max_sentences is not None:
        if args.max_sentences < 10:
            parser.error("--max-sentences must be at least 10")
        all_sentences = all_sentences[: args.max_sentences]

    cfg = load_config(args.config)
    train_sents, val_sents = split_train_val(all_sentences, args.val_fraction, args.seed)
    tags = sorted({tag for _, sentence_tags in all_sentences for tag in sentence_tags})
    vocab = Vocab(train_sents, tag_inventory=tags, case_sensitive_lookup=cfg["case_sensitive_lookup"])
    train_items = [vocab.encode(words, sentence_tags) for words, sentence_tags in train_sents]
    val_items = [vocab.encode(words, sentence_tags) for words, sentence_tags in val_sents]
    device = choose_device(args.device)

    print(f"Device: {device}")
    print(f"Training sentences: {len(train_sents):,}; validation sentences: {len(val_sents):,}")
    print(f"Training tokens: {sum(len(words) for words, _ in train_sents):,}; validation tokens: {sum(len(words) for words, _ in val_sents):,}")
    print(f"Word vocabulary: {len(vocab.word2id):,}; character vocabulary: {len(vocab.char2id):,}; POS tags: {len(vocab.tags)}")
    token_dim = int(cfg["word_dim"]) + int(cfg["shape_dim"]) + (int(cfg["char_out"]) if cfg["char_type"] != "none" else 0)
    print(f"Input vector dimension: {token_dim}/100; decoder: {cfg['decoder']}; max epochs: {cfg['epochs']}")

    model, info = train_model(
        cfg,
        train_items,
        vocab,
        seed=args.seed,
        device=device,
        val_items=val_items,
        max_seconds=args.time_budget,
    )
    # Record the actual epoch budget to make the checkpoint reproducible.
    saved_cfg = dict(cfg)
    saved_cfg["epochs"] = max(1, int(info["best_epoch"]))
    metadata = {
        "selected_by": "held-out training validation accuracy, tie-broken by validation loss",
        "seed": args.seed,
        "training_file": str(data_path.resolve()),
        "training_sentences_total": len(all_sentences),
        "optimization_sentences": len(train_sents),
        "validation_sentences": len(val_sents),
        "best_epoch": info["best_epoch"],
        "best_validation_accuracy": info["best_val_accuracy"],
        "best_validation_loss": info["best_val_loss"],
        "epochs_run": info["epochs_run"],
        "parameters": info["params"],
        "training_seconds": info["seconds"],
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "note": "The saved model uses the best validation checkpoint; run run_all.py to execute the complete ablation suite and full-data refit.",
    }
    save_checkpoint(args.model_out, model, vocab, saved_cfg, metadata)
    print(f"Trainable parameters: {info['params']:,}")
    print(f"Best validation accuracy: {float(info['best_val_accuracy'] or 0) * 100:.3f}% at epoch {info['best_epoch']}")
    print(f"Training time: {info['seconds']:.1f}s; total build time: {time.perf_counter() - started:.1f}s")
    print(f"Saved checkpoint: {Path(args.model_out).resolve()}")


if __name__ == "__main__":
    main()
