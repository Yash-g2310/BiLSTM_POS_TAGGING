#!/usr/bin/env python3
"""Train/forward/backward/checkpoint smoke-stress all registered model variants.

Uses a small subset of the supplied train.tags file (real task data, not synthetic
sentences). This is a software test only; do not report these accuracy numbers.
"""
from __future__ import annotations

import argparse
import os
import json
import tempfile
import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runner import VARIANTS
from tagger_lib import (
    DEFAULT_CFG,
    Tagger,
    Vocab,
    evaluate_model,
    predict_tags,
    read_tagged,
    save_checkpoint,
    split_train_val,
    train_model,
    validate_config,
    load_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--max-sentences", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", default=str(ROOT / "results" / "stress_test.json"))
    parser.add_argument("--variants", default="all", help="comma-separated registered variant names, or all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.max_sentences < 20:
        parser.error("--max-sentences must be at least 20")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    data_dir = Path(args.data_dir)
    all_sents = read_tagged(str(data_dir / "train.tags"))[: args.max_sentences]
    train_sents, val_sents = split_train_val(all_sents, val_fraction=0.20, seed=17)
    global_tags = sorted({tag for _, tags in all_sents for tag in tags})
    records = []
    names = list(VARIANTS) if args.variants == "all" else [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown_names = set(names) - set(VARIANTS)
    if unknown_names:
        parser.error(f"Unknown variant name(s): {sorted(unknown_names)}")
    started = time.perf_counter()
    print(f"Stress test on {len(all_sents)} real training sentences; train={len(train_sents)}, validation={len(val_sents)}, device={device}; variants={names}")
    for run_i, name in enumerate(names, start=1):
        definition = VARIANTS[name]
        cfg = dict(DEFAULT_CFG)
        cfg.update(definition["cfg"])
        cfg["epochs"] = args.epochs
        cfg["patience"] = 1
        cfg["bs"] = min(8, len(train_sents))
        validate_config(cfg)
        vocab = Vocab(train_sents, tag_inventory=global_tags, case_sensitive_lookup=cfg["case_sensitive_lookup"])
        train_items = [vocab.encode(words, tags) for words, tags in train_sents]
        val_items = [vocab.encode(words, tags) for words, tags in val_sents]
        input_dim = cfg["word_dim"] + cfg["shape_dim"] + (cfg["char_out"] if cfg["char_type"] != "none" else 0)
        assert input_dim <= 100, f"{name}: input dimension {input_dim} exceeds PA2 limit"
        print(f"[{run_i}/{len(names)}] {name} (input={input_dim}D, decoder={cfg['decoder']})", flush=True)
        model, info = train_model(cfg, train_items, vocab, seed=23, device=device, val_items=val_items, log=lambda s: print('  ' + s, flush=True))
        metrics = evaluate_model(model, val_items, vocab, cfg, device)
        assert np.isfinite(metrics["accuracy"]), f"{name}: accuracy is not finite"
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert info["params"] > 0
        # Exercise checkpoint serialization / deserialization and prediction.
        with tempfile.TemporaryDirectory(prefix="pos-smoke-") as tmp:
            ckpt_path = Path(tmp) / "tagger_model.pt"
            save_checkpoint(str(ckpt_path), model, vocab, cfg, metadata={"smoke_variant": name})
            loaded, loaded_vocab, loaded_cfg, metadata = load_checkpoint(str(ckpt_path), device)
            smoke_items = [loaded_vocab.encode(item["w"]) for item in val_items[: min(3, len(val_items))]]
            preds = predict_tags(loaded, smoke_items, loaded_vocab, loaded_cfg, device)
            assert len(preds) == len(smoke_items)
            assert all(len(p) == len(item["w"]) for p, item in zip(preds, smoke_items))
            assert metadata.get("smoke_variant") == name
        record = {
            "variant": name,
            "stage": definition["stage"],
            "input_dim": input_dim,
            "decoder": cfg["decoder"],
            "params": info["params"],
            "epochs_run": info["epochs_run"],
            "accuracy_on_smoke_validation_not_for_reporting": metrics["accuracy"],
            "seconds": info["seconds"],
            "status": "PASS",
        }
        records.append(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "PASS", "device": str(device), "data_sentences": len(all_sents), "variants_tested": len(records), "elapsed_seconds": time.perf_counter() - started, "results": records}, indent=2) + "\n", encoding="utf-8")
    print(f"\nPASS: {len(records)}/{len(names)} requested variants trained, evaluated, and round-tripped via checkpoint.")
    print(f"Elapsed: {time.perf_counter() - started:.1f}s; report: {output}")


if __name__ == "__main__":
    main()
