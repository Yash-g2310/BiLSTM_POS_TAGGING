#!/usr/bin/env python3
"""Run the full POS-tagger experiment pipeline end-to-end.

Examples (run these from the project root):
  python run_all.py --profile full --device auto
  python run_all.py --profile standard --device cuda
  python run_all.py --profile full --max-sentences 120 --epochs 1 --smoke-only

The smoke-only option uses a small, real subset of train.tags and intentionally
skips final full-data training. It does not create synthetic training sentences.
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from analysis_tools import build_acl_tex_report, build_report_draft, dump_json, generate_figures, write_dev_diagnostics
from experiment_runner import run_suite
from tagger_lib import (
    DEFAULT_CFG,
    MODEL_FILE,
    Vocab,
    norm,
    evaluate_model,
    make_tagged_lines,
    read_sents,
    read_tagged,
    save_checkpoint,
    train_model,
    validate_config,
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable in this Python environment")
        if requested == "mps" and not getattr(torch.backends, "mps", None):
            raise RuntimeError("--device mps requested but MPS is not available in this PyTorch build")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but torch.backends.mps.is_available() is False")
        device = torch.device(requested)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return device


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="directory containing train.tags, dev.sents, and dev.tags (default: data)")
    ap.add_argument("--output-dir", default="results", help="experiment/results directory (default: results)")
    ap.add_argument("--profile", choices=("quick", "standard", "full"), default="full")
    ap.add_argument("--stage", choices=("all", "ablation", "dims", "size", "reg", "opt", "decoder"), default="all")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs per experiment")
    ap.add_argument("--max-seconds-per-model", type=float, default=None, help="optional soft time limit per variant")
    ap.add_argument("--final-time-budget", type=float, default=840.0, help="soft budget for final full-data fit, seconds")
    ap.add_argument("--max-sentences", type=int, default=None, help="use first N real train.tags sentences; requires --smoke-only")
    ap.add_argument("--smoke-only", action="store_true", help="run model variants on a subset/quick profile and skip deployment fit/dev evaluation")
    ap.add_argument("--resume", action="store_true", help="resume variants recorded in the same results directory")
    args = ap.parse_args()

    if args.max_sentences is not None and not args.smoke_only:
        ap.error("--max-sentences is intended for a smoke run; also pass --smoke-only to prevent a subset model becoming the final model")
    if args.smoke_only and args.max_sentences is None and args.profile != "quick":
        ap.error("For --smoke-only, choose --profile quick or pass --max-sentences N (e.g. 120)")
    if args.final_time_budget <= 0:
        ap.error("--final-time-budget must be positive")

    started = time.perf_counter()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not (data_dir / "train.tags").exists():
        ap.error(f"Missing training file: {data_dir / 'train.tags'}")
    if not (data_dir / "dev.sents").exists():
        ap.error(f"Missing development input: {data_dir / 'dev.sents'}")
    # Fail before spending compute if labelled development files do not align.
    if (data_dir / "dev.tags").exists():
        preflight_sents = read_sents(str(data_dir / "dev.sents"))
        preflight_gold = read_tagged(str(data_dir / "dev.tags"))
        if len(preflight_sents) != len(preflight_gold):
            ap.error(f"dev.sents has {len(preflight_sents)} lines but dev.tags has {len(preflight_gold)} non-empty lines")
        for line_no, (words, (gold_words, _)) in enumerate(zip(preflight_sents, preflight_gold), start=1):
            if words != gold_words:
                ap.error(f"Tokenization mismatch between dev.sents and dev.tags on line {line_no}")

    device = choose_device(args.device)
    print(f"\nPOS tagging experiment pipeline | profile={args.profile} | stage={args.stage} | device={device}")
    print(f"Data: {data_dir}\nResults: {output_dir}")
    suite = run_suite(
        data_dir=data_dir,
        output_dir=output_dir,
        profile=args.profile,
        stage=args.stage,
        device_name=str(device),
        seed=args.seed,
        epochs=args.epochs,
        max_seconds_per_model=args.max_seconds_per_model,
        resume=args.resume,
        max_sentences=args.max_sentences,
    )
    records = suite["records"]
    summary = suite["summary"]
    if not records:
        raise RuntimeError("The experiment suite did not produce any results")
    best = max(records, key=lambda row: float(row["val_accuracy_pct"]))
    selected_cfg = dict(best["cfg"])
    selected_cfg["epochs"] = max(1, int(best.get("best_epoch") or selected_cfg.get("epochs", 1)))
    validate_config(selected_cfg)
    best_config = {
        "best_variant": best["variant"],
        "best_validation_accuracy_pct": best["val_accuracy_pct"],
        "selection_protocol": "highest accuracy on a deterministic sentence-level validation split drawn only from train.tags; dev.tags is not used for selection",
        "cfg": selected_cfg,
    }
    if not args.smoke_only:
        atomic_json(Path("best_config.json"), best_config)
    atomic_json(output_dir / "best_config.json", best_config)
    summary["best_variant"] = best["variant"]
    summary["best_config"] = selected_cfg
    dump_json(output_dir / "experiment_summary.json", summary)

    if args.smoke_only:
        figures = generate_figures(data_dir, output_dir, selected_cfg)
        report_path = build_report_draft(output_dir, selected_cfg, best_variant=best["variant"])
        tex_path = build_acl_tex_report(output_dir, selected_cfg, best_variant=best["variant"])
        print("\nSmoke run complete. No final model was written or selected for deployment.")
        print(f"Variants executed: {len(records)}; best smoke validation score: {best['val_accuracy_pct']:.3f}% (not a publishable result)")
        print(f"Figures: {len(figures)}; report drafts: {report_path} and {tex_path}")
        print(f"Total elapsed: {time.perf_counter() - started:.1f}s")
        return

    # The best epoch is selected on the internal train-derived validation split.
    # We then fit the selected architecture on all of train.tags for deployment.
    train_path = data_dir / "train.tags"
    all_train = read_tagged(str(train_path))
    final_cfg = dict(selected_cfg)
    tags = sorted({tag for _, sentence_tags in all_train for tag in sentence_tags})
    full_vocab = Vocab(all_train, tag_inventory=tags, case_sensitive_lookup=bool(final_cfg.get("case_sensitive_lookup", False)))
    full_items = [full_vocab.encode(words, sentence_tags) for words, sentence_tags in all_train]
    target_epochs = max(1, int(best.get("best_epoch") or selected_cfg["epochs"]))
    final_cfg["epochs"] = target_epochs
    print(f"\nFinal full-data refit: variant={best['variant']}, epochs={target_epochs}, train sentences={len(all_train):,}")
    final_model, final_info = train_model(
        final_cfg,
        full_items,
        full_vocab,
        seed=args.seed,
        device=device,
        val_items=None,
        max_seconds=args.final_time_budget,
    )
    final_cfg["epochs"] = max(1, int(final_info["epochs_run"]))
    save_checkpoint(
        MODEL_FILE,
        final_model,
        full_vocab,
        final_cfg,
        metadata={
            "variant": best["variant"],
            "selected_validation_accuracy_pct": best["val_accuracy_pct"],
            "selected_epoch": target_epochs,
            "full_data_epochs_run": final_info["epochs_run"],
            "parameters": final_info["params"],
            "training_seconds": final_info["seconds"],
            "device": str(device),
            "seed": args.seed,
            "train_sentences": len(all_train),
            "train_tokens": sum(len(words) for words, _ in all_train),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    )
    final_model.eval()
    print(f"Final model saved to {Path(MODEL_FILE).resolve()} ({final_info['params']:,} parameters, {final_info['seconds']:.1f}s)")

    dev_sents_path, dev_tags_path = data_dir / "dev.sents", data_dir / "dev.tags"
    inference_device_arg = device.type
    cli_started = time.perf_counter()
    cli_command = [
        sys.executable, str(Path(__file__).resolve().parent / "run_tagger.py"),
        str(dev_sents_path.resolve()), str(Path("dev.out").resolve()),
        "--model", str(Path(MODEL_FILE).resolve()), "--device", inference_device_arg,
    ]
    subprocess.run(cli_command, check=True)
    run_tagger_wall_seconds = time.perf_counter() - cli_started
    print(f"run_tagger.py wall runtime (process start + checkpoint load + inference): {run_tagger_wall_seconds:.2f}s")

    if not dev_tags_path.exists():
        print(f"WARNING: {dev_tags_path} is absent. Skipping labeled dev evaluation; inference on dev.sents will still be possible.")
        dev_sents = read_sents(str(dev_sents_path))
        nonempty = [words for words in dev_sents if words]
        nonempty_items = [full_vocab.encode(words) for words in nonempty]
        from tagger_lib import predict_tags
        preds = predict_tags(final_model, nonempty_items, full_vocab, final_cfg, device)
        tagged_nonempty = make_tagged_lines(nonempty, preds, full_vocab.tags)
        iterator = iter(tagged_nonempty)
        lines = [next(iterator) if words else "" for words in dev_sents]
        Path("dev.out").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        dev_sents = read_sents(str(dev_sents_path))
        gold_sents = read_tagged(str(dev_tags_path))
        if len(dev_sents) != len(gold_sents):
            raise ValueError(f"dev.sents has {len(dev_sents)} lines but dev.tags has {len(gold_sents)} non-empty lines")
        for i, (words, (gold_words, _)) in enumerate(zip(dev_sents, gold_sents), start=1):
            if words != gold_words:
                raise ValueError(f"Tokenization mismatch between dev.sents and dev.tags on line {i}")
        dev_items = [full_vocab.encode(words, gold_tags) for words, gold_tags in gold_sents]
        dev_eval = evaluate_model(final_model, dev_items, full_vocab, final_cfg, device, include_predictions=True)
        pred_indices = dev_eval.pop("predictions")
        pred_tags = [[full_vocab.tags[int(tag_id)] for tag_id in row] for row in pred_indices]
        tagged_lines = make_tagged_lines(dev_sents, pred_indices, full_vocab.tags)
        (output_dir / "dev.out").write_text(Path("dev.out").read_text(encoding="utf-8"), encoding="utf-8")
        cli_output = read_tagged("dev.out")
        if len(cli_output) != len(tagged_lines):
            raise RuntimeError("run_tagger.py output sentence count differs from programmatic evaluation")
        for i, (cli_pair, expected_line) in enumerate(zip(cli_output, tagged_lines), start=1):
            if " ".join(f"{w}/{t}" for w, t in zip(*cli_pair)) != expected_line:
                raise RuntimeError(f"run_tagger.py output differs from programmatic predictions on sentence {i}")
        # Save metrics from the same model and independently computed diagnostic tables.
        dump_json(output_dir / "dev_model_loss_metrics.json", dev_eval)
        diagnostics = write_dev_diagnostics(
            dev_sents=dev_sents,
            gold_sents=gold_sents,
            pred_tags=pred_tags,
            training_surface_words=full_vocab.surface_words,
            normalized_vocab={norm(word) for words, _ in all_train for word in words},
            lookup_vocab=set(full_vocab.word2id) - {"<pad>", "<unk>"},
            case_sensitive_lookup=bool(final_cfg.get("case_sensitive_lookup", False)),
            output_dir=output_dir,
        )
        print("\nDevelopment set evaluation (not used for model selection):")
        print(f"  Token accuracy: {diagnostics['token_accuracy'] * 100:.3f}% ({diagnostics['correct_tokens']:,}/{diagnostics['token_count']:,})")
        print(f"  Exact-sentence accuracy: {diagnostics['exact_sentence_accuracy'] * 100:.3f}%")
        if diagnostics["surface_oov_accuracy"] is not None:
            print(f"  Exact-surface OOV accuracy: {diagnostics['surface_oov_accuracy'] * 100:.3f}% ({diagnostics['surface_oov_count']:,} tokens)")
        if diagnostics["normalized_oov_accuracy"] is not None:
            print(f"  Lowercase/digit-normalized OOV accuracy: {diagnostics['normalized_oov_accuracy'] * 100:.3f}% ({diagnostics['normalized_oov_count']:,} tokens)")
        if diagnostics["lookup_oov_accuracy"] is not None:
            print(f"  Selected-model lexical-lookup OOV accuracy: {diagnostics['lookup_oov_accuracy'] * 100:.3f}% ({diagnostics['lookup_oov_count']:,} tokens)")
        print(f"  Wrote {Path('dev.out').resolve()}")

    # Persist measured deployment timings before rendering the report so the report
    # contains the actual selected-model refit and run_tagger.py runtimes.
    summary["final_model_runtime"] = {
        "final_fit_seconds": round(float(final_info["seconds"]), 2),
        "final_fit_epochs": int(final_info["epochs_run"]),
        "final_parameters": int(final_info["params"]),
        "run_tagger_wall_seconds": round(float(run_tagger_wall_seconds), 2),
        "dev_tokens": int(sum(len(sentence) for sentence in dev_sents)),
        "device": str(device),
    }
    dump_json(output_dir / "experiment_summary.json", summary)

    figures = generate_figures(data_dir, output_dir, final_cfg)
    report_path = build_report_draft(output_dir, final_cfg, best_variant=best["variant"])
    tex_path = build_acl_tex_report(output_dir, final_cfg, best_variant=best["variant"])
    runtime = {
        "pipeline_total_seconds": time.perf_counter() - started,
        "experiment_variant_count": len(records),
        "best_variant": best["variant"],
        "best_validation_accuracy_pct": best["val_accuracy_pct"],
        "final_fit_seconds": final_info["seconds"],
        "run_tagger_wall_seconds": run_tagger_wall_seconds,
        "final_fit_epochs": final_info["epochs_run"],
        "final_parameters": final_info["params"],
        "device": str(device),
        "model_file": str(Path(MODEL_FILE).resolve()),
        "dev_output_file": str(Path("dev.out").resolve()),
        "figures": figures,
        "report_draft": str(report_path.resolve()),
        "acl_tex_draft": str(tex_path.resolve()),
    }
    dump_json(output_dir / "runtime_summary.json", runtime)
    print(f"\nPipeline complete in {runtime['pipeline_total_seconds'] / 60:.1f} minutes.")
    print(f"Best internal validation variant: {best['variant']} ({best['val_accuracy_pct']:.3f}%)")
    print(f"Generated {len(figures)} plots/diagrams and report drafts: {report_path}, {tex_path}")
    print(f"CSV results: {output_dir / 'experiments.csv'}")
    print(f"Error-analysis tables: {output_dir / 'per_tag_metrics.csv'}, {output_dir / 'confusion_pairs.csv'}, {output_dir / 'error_examples.csv'}")


if __name__ == "__main__":
    main()
