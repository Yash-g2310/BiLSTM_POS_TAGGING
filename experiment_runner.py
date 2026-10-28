"""Experiment definitions and shared runner for PA2 POS-tagger ablations."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tagger_lib import (
    DEFAULT_CFG,
    MODEL_FILE,
    Tagger,
    Vocab,
    evaluate_model,
    load_checkpoint,
    make_tagged_lines,
    n_params,
    predict_tags,
    read_sents,
    read_tagged,
    save_checkpoint,
    split_train_val,
    train_model,
    validate_config,
)

# Every architecture variant stays within the assignment's 100-D token-vector limit.
VARIANTS: dict[str, dict[str, Any]] = {
    # Core architecture ablations.
    "word_only": {"stage": "ablation", "description": "100-D word embeddings + BiLSTM", "cfg": {"char_type": "none", "word_dim": 100, "char_out": 0, "shape_dim": 0}},
    "word_plus_shape": {"stage": "ablation", "description": "90-D word embeddings + 10-D shape embedding + BiLSTM", "cfg": {"char_type": "none", "word_dim": 90, "char_out": 0, "shape_dim": 10}},
    "word_char_cnn": {"stage": "ablation", "description": "60-D word + 40-D character CNN; no shape embedding", "cfg": {"word_dim": 60, "char_out": 40, "shape_dim": 0, "char_type": "cnn"}},
    "full_char_cnn_bilstm": {"stage": "ablation", "description": "50-D word + 40-D character CNN + 10-D shape; BiLSTM + softmax", "cfg": {}},
    "full_char_lstm_bilstm": {"stage": "ablation", "description": "Character LSTM instead of character CNN", "cfg": {"char_type": "lstm"}},
    "case_sensitive_lookup": {"stage": "ablation", "description": "Preserve lexical case instead of lowercasing (digits still normalized)", "cfg": {"case_sensitive_lookup": True}},
    "forward_only": {"stage": "ablation", "description": "Unidirectional LSTM instead of bidirectional LSTM", "cfg": {"bidirectional": False}},
    "full_char_cnn_crf": {"stage": "decoder", "description": "Same encoder with a linear-chain CRF decoder", "cfg": {"decoder": "crf"}},
    # Feature-budget / representation ablations.
    "w80_c10_s10": {"stage": "dims", "description": "80-D word + 10-D character + 10-D shape", "cfg": {"word_dim": 80, "char_out": 10, "shape_dim": 10}},
    "paper_spec_80_20_192": {"stage": "dims", "description": "Peer-report-like 80-D word + 20-D multi-width character CNN; 192-unit BiLSTM and 16-D char embeddings", "cfg": {"word_dim": 80, "char_out": 20, "shape_dim": 0, "hidden": 192, "char_emb": 16}},
    "w60_c30_s10": {"stage": "dims", "description": "60-D word + 30-D character + 10-D shape", "cfg": {"word_dim": 60, "char_out": 30, "shape_dim": 10}},
    "w40_c50_s10": {"stage": "dims", "description": "40-D word + 50-D character + 10-D shape", "cfg": {"word_dim": 40, "char_out": 50, "shape_dim": 10}},
    # Capacity / regularisation / optimization ablations.
    "hidden128": {"stage": "size", "description": "128 hidden units per recurrent direction", "cfg": {"hidden": 128}},
    "hidden384": {"stage": "size", "description": "384 hidden units per recurrent direction", "cfg": {"hidden": 384}},
    "one_lstm_layer": {"stage": "size", "description": "One recurrent layer rather than two", "cfg": {"layers": 1}},
    "dropout_low": {"stage": "reg", "description": "Lower dropout: input/output 0.25, recurrent 0.20", "cfg": {"emb_drop": 0.25, "lstm_drop": 0.20, "out_drop": 0.25}},
    "dropout_high": {"stage": "reg", "description": "Higher dropout: input/output 0.50, recurrent 0.40", "cfg": {"emb_drop": 0.50, "lstm_drop": 0.40, "out_drop": 0.50}},
    "no_word_dropout": {"stage": "reg", "description": "Disable frequency-aware word dropout", "cfg": {"wdrop_alpha": 0.0}},
    "word_dropout_half": {"stage": "reg", "description": "Reduce frequency-aware word-drop strength", "cfg": {"wdrop_alpha": 0.5}},
    "lr_1e3": {"stage": "opt", "description": "AdamW learning rate 0.001", "cfg": {"lr": 1e-3}},
    "lr_3e3": {"stage": "opt", "description": "AdamW learning rate 0.003", "cfg": {"lr": 3e-3}},
    "batch128": {"stage": "opt", "description": "Batch size 128 sentences", "cfg": {"bs": 128}},
    "label_smoothing_01": {"stage": "reg", "description": "Cross-entropy label smoothing 0.1", "cfg": {"label_smooth": 0.1}},
}

PROFILE_VARIANTS = {
    "quick": ["word_only", "word_char_cnn", "full_char_cnn_bilstm", "full_char_lstm_bilstm"],
    "standard": [
        "word_only", "word_plus_shape", "word_char_cnn", "full_char_cnn_bilstm",
        "full_char_lstm_bilstm", "forward_only", "full_char_cnn_crf", "w80_c10_s10",
        "w40_c50_s10", "hidden128", "dropout_low", "lr_1e3",
    ],
    "full": list(VARIANTS.keys()),
}


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable in this Python environment")
        device = torch.device(requested)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _save_results_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "variant", "stage", "description", "seed", "val_accuracy_pct", "val_surface_oov_accuracy_pct",
        "val_lookup_oov_accuracy_pct", "val_exact_sentence_pct", "val_loss", "val_tokens",
        "val_surface_oov_tokens", "params", "epochs_run", "best_epoch", "seconds", "input_dim", "cfg_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "cfg_json": json.dumps(record.get("cfg", {}), sort_keys=True)})


def _base_cfg(name: str, overrides: dict[str, Any], epochs: int) -> dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    cfg["epochs"] = epochs
    cfg.update(overrides)
    validate_config(cfg)
    return cfg


def run_suite(
    *,
    data_dir: str | Path = ".",
    output_dir: str | Path = "results",
    profile: str = "standard",
    stage: str = "all",
    device_name: str = "auto",
    seed: int = 42,
    epochs: int | None = None,
    max_seconds_per_model: float | None = None,
    resume: bool = False,
    max_sentences: int | None = None,
) -> dict[str, Any]:
    """Run requested variant set on a common sentence-level 95/5 split.

    Reports only held-out-from-training validation results for experiment selection.
    If a labelled dev.tags file exists, run_all.py evaluates it after model selection;
    dev labels are never used to choose hyperparameters.
    """
    data_path = Path(data_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = data_path / "train.tags"
    dev_sents_path = data_path / "dev.sents"
    if not train_path.exists():
        raise FileNotFoundError(f"Required file not found: {train_path}")
    if not dev_sents_path.exists():
        raise FileNotFoundError(f"Required file not found: {dev_sents_path}")
    if stage == "all":
        variant_names = PROFILE_VARIANTS[profile]
    else:
        if stage not in {"ablation", "dims", "size", "reg", "opt", "decoder"}:
            raise ValueError(f"Unsupported stage {stage!r}; choose all/ablation/dims/size/reg/opt/decoder")
        variant_names = [name for name, item in VARIANTS.items() if item["stage"] == stage]
        if not variant_names:
            raise ValueError(f"No variants registered for stage {stage!r}")
    epochs_each = epochs if epochs is not None else (4 if profile == "quick" else 18 if profile == "full" else 16)
    device = choose_device(device_name)
    print(f"Device: {device}; stage={stage}; profile={profile}; variants={len(variant_names)}; max epochs={epochs_each}")
    print(f"Dataset directory: {data_path}")
    print(f"Output directory:  {output_path}")

    all_sentences = read_tagged(str(train_path))
    if max_sentences is not None:
        if max_sentences < 10:
            raise ValueError("max_sentences must be at least 10 so both train and validation splits are meaningful")
        all_sentences = all_sentences[:max_sentences]
    train_sents, val_sents = split_train_val(all_sentences, val_fraction=0.05, seed=seed)
    tag_inventory = sorted({tag for _, tags in all_sentences for tag in tags})
    vocab = Vocab(train_sents, tag_inventory=tag_inventory)
    print(
        f"Data: {len(all_sentences):,} sentences / {sum(len(w) for w, _ in all_sentences):,} tokens; "
        f"train={len(train_sents):,}; validation={len(val_sents):,}; "
        f"word vocab={len(vocab.word2id):,}; char vocab={len(vocab.char2id):,}; tags={len(vocab.tags)}"
    )

    data_fingerprints = {"train.tags": _sha256(train_path), "dev.sents": _sha256(dev_sents_path)}
    if (data_path / "dev.tags").exists():
        data_fingerprints["dev.tags"] = _sha256(data_path / "dev.tags")
    manifest_path = output_path / "run_manifest.json"
    run_config = {
        "profile": profile,
        "stage": stage,
        "variant_names": variant_names,
        "seed": seed,
        "epochs_each": epochs_each,
        "max_sentences": max_sentences,
        "val_fraction": 0.05,
        "data_fingerprints": data_fingerprints,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "device": str(device),
    }
    runs_path = output_path / "experiment_runs.jsonl"
    if resume and manifest_path.exists():
        existing_config = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("profile", "stage", "variant_names", "seed", "epochs_each", "max_sentences", "data_fingerprints"):
            if existing_config.get(key) != run_config.get(key):
                raise ValueError(f"Cannot resume: run manifest differs in {key}. Use a fresh output directory or --no-resume.")
    elif manifest_path.exists() and not resume:
        # Preserve the previous results instead of silently appending incompatible runs.
        archive = output_path / f"previous_run_{int(time.time())}"
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("run_manifest.json", "experiment_runs.jsonl", "experiments.csv", "history.csv"):
            old = output_path / name
            if old.exists():
                old.replace(archive / name)
        best_ckpt = output_path / "checkpoints" / "selection_best.pt"
        if best_ckpt.exists():
            best_ckpt.replace(archive / "selection_best.pt")
    _write_json(manifest_path, run_config)
    records = _load_jsonl(runs_path) if resume else []
    completed = {record["variant"] for record in records}
    vocab_cache: dict[bool, tuple[Vocab, list[dict[str, Any]], list[dict[str, Any]]]] = {}

    def prepare_variant_data(case_sensitive: bool) -> tuple[Vocab, list[dict[str, Any]], list[dict[str, Any]]]:
        if case_sensitive not in vocab_cache:
            variant_vocab = Vocab(train_sents, tag_inventory=tag_inventory, case_sensitive_lookup=case_sensitive)
            variant_train = [variant_vocab.encode(words, tags) for words, tags in train_sents]
            variant_val = [variant_vocab.encode(words, tags) for words, tags in val_sents]
            vocab_cache[case_sensitive] = (variant_vocab, variant_train, variant_val)
        return vocab_cache[case_sensitive]

    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_score = max((float(record["val_accuracy_pct"]) for record in records), default=-1.0)
    best_variant = next((record["variant"] for record in records if float(record["val_accuracy_pct"]) == best_score), None)
    best_ckpt = checkpoint_dir / "selection_best.pt"

    for run_number, name in enumerate(variant_names, start=1):
        if name in completed:
            print(f"[{run_number}/{len(variant_names)}] Skipping completed {name} (--resume)")
            continue
        definition = VARIANTS[name]
        cfg = _base_cfg(name, definition["cfg"], epochs_each)
        variant_vocab, train_items, val_items = prepare_variant_data(cfg["case_sensitive_lookup"])
        print(f"\n[{run_number}/{len(variant_names)}] {name}: {definition['description']}")
        print(f"  token vector={cfg['word_dim'] + (cfg['char_out'] if cfg['char_type'] != 'none' else 0) + cfg['shape_dim']}D; decoder={cfg['decoder']}; case-sensitive={cfg['case_sensitive_lookup']}")
        model, info = train_model(
            cfg, train_items, variant_vocab, seed=seed, device=device, val_items=val_items,
            max_seconds=max_seconds_per_model,
            log=lambda message: print(message, flush=True),
        )
        metrics = evaluate_model(model, val_items, variant_vocab, cfg, device)
        record = {
            "variant": name,
            "stage": definition["stage"],
            "description": definition["description"],
            "seed": seed,
            "val_accuracy_pct": round(metrics["accuracy"] * 100, 5),
            "val_surface_oov_accuracy_pct": round(metrics["surface_oov_accuracy"] * 100, 5) if np.isfinite(metrics["surface_oov_accuracy"]) else None,
            "val_lookup_oov_accuracy_pct": round(metrics["lookup_oov_accuracy"] * 100, 5) if np.isfinite(metrics["lookup_oov_accuracy"]) else None,
            "val_exact_sentence_pct": round(metrics["exact_sentence_accuracy"] * 100, 5),
            "val_loss": round(metrics["token_loss"], 6),
            "val_tokens": metrics["tokens"],
            "val_surface_oov_tokens": metrics["surface_oov_tokens"],
            "params": info["params"],
            "epochs_run": info["epochs_run"],
            "best_epoch": info["best_epoch"],
            "seconds": round(info["seconds"], 2),
            "input_dim": model.input_dim,
            "vocab_size": len(variant_vocab.word2id),
            "case_sensitive_lookup": variant_vocab.case_sensitive_lookup,
            "cfg": cfg,
            "history": info["history"],
        }
        records.append(record)
        _save_jsonl(runs_path, records)
        _save_results_csv(output_path / "experiments.csv", records)
        if record["val_accuracy_pct"] > best_score:
            best_score = record["val_accuracy_pct"]
            best_variant = name
            save_checkpoint(
                str(best_ckpt), model, variant_vocab, cfg,
                metadata={"variant": name, "selection_metrics": record, "run_config": run_config},
            )
        print(
            f"  RESULT val={record['val_accuracy_pct']:.3f}% "
            f"surface-OOV={record['val_surface_oov_accuracy_pct']}% "
            f"params={record['params']:,} best_epoch={record['best_epoch']} time={record['seconds']:.1f}s"
        )

    _save_results_csv(output_path / "experiments.csv", records)
    history_rows: list[dict[str, Any]] = []
    for record in records:
        for entry in record.get("history", []):
            history_rows.append({"variant": record["variant"], **entry})
    if history_rows:
        keys = sorted({key for row in history_rows for key in row})
        with (output_path / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history_rows)
    summary = {
        "run_config": run_config,
        "train_sentences_total": len(all_sentences),
        "max_sentences_for_run": max_sentences,
        "train_tokens_total": sum(len(words) for words, _ in all_sentences),
        "optimization_sentences": len(train_sents),
        "validation_sentences": len(val_sents),
        "validation_tokens": sum(len(words) for words, _ in val_sents),
        "validation_vocab_size": len(vocab.word2id),
        "character_vocab_size": len(vocab.char2id),
        "tag_count": len(vocab.tags),
        "completed_variants": [record["variant"] for record in records],
        "best_variant": best_variant,
        "best_validation_accuracy_pct": best_score,
        "dev_tags_available": (data_path / "dev.tags").exists(),
        "note": "All variant scores are from the training-derived 5% sentence holdout. dev.sents labels are not used for selection.",
    }
    _write_json(output_path / "experiment_summary.json", summary)
    return {
        "records": records,
        "summary": summary,
        "best_checkpoint": best_ckpt,
        "data_path": data_path,
        "output_path": output_path,
        "device": device,
    }
