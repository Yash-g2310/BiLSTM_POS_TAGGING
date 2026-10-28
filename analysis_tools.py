"""Post-training diagnostics, plots, and a draft research-style report for PA2."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tagger_lib import lexical_key, norm, read_sents, read_tagged


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_csv(path: Path, records: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(key for row in records for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def validate_parallel_data(sents: Sequence[Sequence[str]], tagged_sents: Sequence[tuple[Sequence[str], Sequence[str]]]) -> None:
    if len(sents) != len(tagged_sents):
        raise ValueError(f"Sentence count mismatch: {len(sents)} input vs {len(tagged_sents)} gold")
    for i, (words, (gold_words, tags)) in enumerate(zip(sents, tagged_sents), start=1):
        if list(words) != list(gold_words):
            raise ValueError(f"Tokenization mismatch in sentence {i}: input has {len(words)} tokens, gold has {len(gold_words)}")
        if len(words) != len(tags):
            raise ValueError(f"Gold word/tag count mismatch in sentence {i}")


def write_dev_diagnostics(
    *,
    dev_sents: Sequence[Sequence[str]],
    gold_sents: Sequence[tuple[Sequence[str], Sequence[str]]],
    pred_tags: Sequence[Sequence[str]],
    training_surface_words: set[str],
    normalized_vocab: set[str],
    output_dir: str | Path,
    lookup_vocab: set[str] | None = None,
    case_sensitive_lookup: bool = False,
) -> dict[str, Any]:
    """Write per-token predictions and error analysis; all numbers are computed from supplied data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    validate_parallel_data(dev_sents, gold_sents)
    if len(pred_tags) != len(dev_sents):
        raise ValueError(f"Prediction sentence count mismatch: {len(pred_tags)} vs {len(dev_sents)}")

    correct = total = sentence_exact = 0
    surface_oov_total = surface_oov_correct = 0
    normalized_oov_total = normalized_oov_correct = 0
    lookup_oov_total = lookup_oov_correct = 0
    lookup_vocab = normalized_vocab if lookup_vocab is None else lookup_vocab
    tag_counts: Counter[str] = Counter()
    tag_correct: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    token_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    bucket_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for sent_i, (words, (gold_words, gold), pred) in enumerate(zip(dev_sents, gold_sents, pred_tags), start=1):
        if len(pred) != len(words):
            raise ValueError(f"Token/tag count mismatch in sentence {sent_i}: {len(words)} words vs {len(pred)} tags")
        sent_correct = 0
        for tok_i, (word, true_tag, predicted_tag) in enumerate(zip(words, gold, pred), start=1):
            hit = true_tag == predicted_tag
            total += 1
            correct += int(hit)
            sent_correct += int(hit)
            tag_counts[true_tag] += 1
            tag_correct[true_tag] += int(hit)
            pred_counts[predicted_tag] += 1
            if not hit:
                confusion[(true_tag, predicted_tag)] += 1
            surface_oov = word not in training_surface_words
            normalized_oov = norm(word) not in normalized_vocab
            lookup_oov = lexical_key(word, case_sensitive_lookup) not in lookup_vocab
            if surface_oov:
                surface_oov_total += 1
                surface_oov_correct += int(hit)
            if normalized_oov:
                normalized_oov_total += 1
                normalized_oov_correct += int(hit)
            if lookup_oov:
                lookup_oov_total += 1
                lookup_oov_correct += int(hit)
            row = {
                "sentence_id": sent_i,
                "token_id": tok_i,
                "word": word,
                "gold_tag": true_tag,
                "predicted_tag": predicted_tag,
                "correct": int(hit),
                "surface_oov": int(surface_oov),
                "normalized_oov": int(normalized_oov),
                "model_lookup_oov": int(lookup_oov),
            }
            token_rows.append(row)
            if not hit:
                left = max(0, tok_i - 4)
                right = min(len(words), tok_i + 3)
                context = " ".join(words[left:right])
                error_rows.append({**row, "context": context})
        sentence_exact += int(sent_correct == len(words))
        bucket = "1-10" if len(words) <= 10 else "11-20" if len(words) <= 20 else "21-30" if len(words) <= 30 else "31-50" if len(words) <= 50 else "51+"
        bucket_stats[bucket][0] += sent_correct
        bucket_stats[bucket][1] += len(words)
        length_rows.append({"sentence_id": sent_i, "tokens": len(words), "correct_tokens": sent_correct, "accuracy": sent_correct / max(1, len(words))})

    per_tag_rows = []
    for tag in sorted(tag_counts, key=lambda x: (-tag_counts[x], x)):
        support = tag_counts[tag]
        tp = tag_correct[tag]
        precision = tp / pred_counts[tag] if pred_counts[tag] else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_tag_rows.append({"tag": tag, "support": support, "correct": tp, "errors": support - tp, "accuracy_recall": recall, "precision": precision, "f1": f1})
    confusion_rows = [
        {"gold_tag": gold, "predicted_tag": pred, "count": count}
        for (gold, pred), count in confusion.most_common()
    ]
    length_bins = [
        {"sentence_length_bucket": bucket, "tokens": vals[1], "correct_tokens": vals[0], "accuracy": vals[0] / vals[1] if vals[1] else float("nan")}
        for bucket, vals in bucket_stats.items()
    ]
    metrics = {
        "dataset": "provided development set",
        "sentence_count": len(dev_sents),
        "token_count": total,
        "correct_tokens": correct,
        "token_accuracy": correct / total if total else float("nan"),
        "token_error_rate": (total - correct) / total if total else float("nan"),
        "exact_sentence_count": sentence_exact,
        "exact_sentence_accuracy": sentence_exact / len(dev_sents) if dev_sents else float("nan"),
        "surface_oov_count": surface_oov_total,
        "surface_oov_correct": surface_oov_correct,
        "surface_oov_accuracy": surface_oov_correct / surface_oov_total if surface_oov_total else None,
        "normalized_oov_count": normalized_oov_total,
        "normalized_oov_correct": normalized_oov_correct,
        "normalized_oov_accuracy": normalized_oov_correct / normalized_oov_total if normalized_oov_total else None,
        "lookup_oov_count": lookup_oov_total,
        "lookup_oov_correct": lookup_oov_correct,
        "lookup_oov_accuracy": lookup_oov_correct / lookup_oov_total if lookup_oov_total else None,
        "known_surface_count": total - surface_oov_total,
        "known_surface_accuracy": (correct - surface_oov_correct) / (total - surface_oov_total) if total > surface_oov_total else None,
        "tag_count": len(tag_counts),
        "definition_surface_oov": "Exact token surface form does not occur anywhere in train.tags.",
        "definition_normalized_oov": "Lowercased, digit-normalized form is absent from the training corpus lexical forms, independent of the selected lookup mode.",
        "definition_lookup_oov": "The lexical key used by the selected model is absent from its training vocabulary (case-aware if configured).",
    }
    dump_json(out / "dev_metrics.json", metrics)
    _write_csv(out / "dev_token_predictions.csv", token_rows)
    _write_csv(out / "per_tag_metrics.csv", per_tag_rows)
    _write_csv(out / "confusion_pairs.csv", confusion_rows)
    _write_csv(out / "error_examples.csv", error_rows)
    _write_csv(out / "sentence_accuracy.csv", length_rows)
    _write_csv(out / "length_bucket_metrics.csv", length_bins)
    return {**metrics, "per_tag": per_tag_rows, "confusions": confusion_rows, "length_buckets": length_bins}


def plot_architecture(cfg: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dimensions = int(cfg["word_dim"]) + int(cfg["shape_dim"]) + (int(cfg["char_out"]) if cfg["char_type"] != "none" else 0)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.2, 2.55, 1.45, 1.05, "Tokenized\nsentence", "#edf2f7"),
        (2.0, 4.0, 2.0, 0.9, f"Word embedding\n{cfg['word_dim']}D", "#d9eaf7"),
        (2.0, 2.55, 2.0, 0.9, f"Character {cfg['char_type'].upper()}\n{cfg['char_out'] if cfg['char_type'] != 'none' else 0}D" + ("\nwidths 3,5" if cfg['char_type'] == 'cnn' else ""), "#e3f2dd"),
        (2.0, 1.1, 2.0, 0.9, f"Shape embedding\n{cfg['shape_dim']}D", "#fff0d9"),
        (4.45, 2.55, 1.65, 1.05, f"Concatenate + dropout\n{dimensions}D", "#f3e4f7"),
        (6.55, 2.55, 1.8, 1.05, f"{cfg['layers']}-layer " + ("BiLSTM" if cfg["bidirectional"] else "LSTM") + f"\n{cfg['hidden']} units/direction", "#dce8fb"),
        (8.8, 2.55, 1.45, 1.05, f"Linear layer\n{len(cfg.get('tags', [])) or 45} tags", "#fbe3df"),
        (10.65, 2.55, 1.15, 1.05, cfg["decoder"].upper() + "\nDecoder", "#edf2f7"),
    ]
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.09", linewidth=1.1, edgecolor="#44546a", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.3, wrap=True)
    def arrow(a, b, c, d):
        ax.add_patch(FancyArrowPatch((a, b), (c, d), arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#44546a"))
    arrow(1.65, 3.08, 2.0, 4.42)
    arrow(1.65, 3.08, 2.0, 3.0)
    arrow(1.65, 3.08, 2.0, 1.55)
    arrow(4.0, 4.42, 4.45, 3.45)
    arrow(4.0, 3.0, 4.45, 3.08)
    arrow(4.0, 1.55, 4.45, 2.73)
    arrow(6.1, 3.08, 6.55, 3.08)
    arrow(8.35, 3.08, 8.8, 3.08)
    arrow(10.25, 3.08, 10.65, 3.08)
    ax.text(6, 5.45, "Neural POS tagger — selected configuration", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(6, 0.35, "No pretrained vectors • token input representation constrained to ≤100 dimensions", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "architecture.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "architecture.svg", bbox_inches="tight")
    plt.close(fig)


def generate_figures(data_dir: str | Path, output_dir: str | Path, cfg: dict[str, Any]) -> list[str]:
    data_path, root = Path(data_dir), Path(output_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    plot_architecture(cfg, figures)
    generated += ["figures/architecture.png", "figures/architecture.svg"]

    exp_csv = root / "experiments.csv"
    if exp_csv.exists():
        import pandas as pd
        exp = pd.read_csv(exp_csv)
        if not exp.empty and "val_accuracy_pct" in exp.columns:
            exp = exp.sort_values("val_accuracy_pct", ascending=True)
            height = max(5.2, 0.34 * len(exp))
            fig, ax = plt.subplots(figsize=(10, height))
            labels = exp["variant"].astype(str).tolist()
            values = exp["val_accuracy_pct"].astype(float).to_numpy()
            ax.barh(labels, values)
            ax.set_xlabel("Held-out training validation accuracy (%)")
            ax.set_title("Ablation and hyperparameter comparison")
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            fig.tight_layout()
            fig.savefig(figures / "experiment_accuracy.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated.append("figures/experiment_accuracy.png")
            if "val_surface_oov_accuracy_pct" in exp.columns:
                valid = exp.dropna(subset=["val_surface_oov_accuracy_pct"])
                if not valid.empty:
                    fig, ax = plt.subplots(figsize=(8, 5.4))
                    ax.scatter(valid["val_surface_oov_accuracy_pct"], valid["val_accuracy_pct"])
                    for _, row in valid.iterrows():
                        ax.annotate(str(row["variant"]), (row["val_surface_oov_accuracy_pct"], row["val_accuracy_pct"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
                    ax.set_xlabel("Validation accuracy on exact-surface OOV tokens (%)")
                    ax.set_ylabel("Overall validation accuracy (%)")
                    ax.set_title("Overall versus unseen-surface token accuracy")
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    fig.savefig(figures / "overall_vs_oov.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    generated.append("figures/overall_vs_oov.png")

    history_csv = root / "history.csv"
    if history_csv.exists():
        import pandas as pd
        hist = pd.read_csv(history_csv)
        desired = ["full_char_cnn_bilstm", "word_only", "full_char_cnn_crf"]
        available = [name for name in desired if "variant" in hist and name in set(hist["variant"].astype(str))]
        if available and "val_accuracy" in hist.columns:
            fig, ax = plt.subplots(figsize=(8, 5.2))
            for name in available:
                sub = hist[hist["variant"].astype(str) == name].sort_values("epoch")
                ax.plot(sub["epoch"], sub["val_accuracy"] * 100, marker="o", markersize=3, label=name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation accuracy (%)")
            ax.set_title("Learning curves (training-derived holdout)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures / "learning_curves.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated.append("figures/learning_curves.png")

    per_tag_csv = root / "per_tag_metrics.csv"
    if per_tag_csv.exists():
        import pandas as pd
        per_tag = pd.read_csv(per_tag_csv)
        if not per_tag.empty:
            worst = per_tag.sort_values("accuracy_recall", ascending=True).head(16).sort_values("accuracy_recall", ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(4.4, 0.3 * len(worst))))
            ax.barh(worst["tag"], worst["accuracy_recall"] * 100)
            ax.set_xlabel("Tag-level recall / accuracy (%)")
            ax.set_title("Lowest-accuracy POS tags on the development set")
            ax.set_xlim(0, 100)
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures / "per_tag_error_rates.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated.append("figures/per_tag_error_rates.png")

    confusion_csv = root / "confusion_pairs.csv"
    if confusion_csv.exists():
        import pandas as pd
        pairs = pd.read_csv(confusion_csv)
        pairs = pairs[pairs["gold_tag"] != pairs["predicted_tag"]].head(15)
        if not pairs.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            labels = [f"{a} → {b}" for a, b in zip(pairs["gold_tag"], pairs["predicted_tag"])]
            ax.barh(labels[::-1], pairs["count"].astype(int).tolist()[::-1])
            ax.set_xlabel("Number of errors")
            ax.set_title("Most frequent directed POS-tag confusions")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures / "tag_confusions.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated.append("figures/tag_confusions.png")

    length_csv = root / "length_bucket_metrics.csv"
    if length_csv.exists():
        import pandas as pd
        bins = pd.read_csv(length_csv)
        if not bins.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            ax.bar(bins["sentence_length_bucket"].astype(str), bins["accuracy"].astype(float) * 100)
            ax.set_xlabel("Sentence length (tokens)")
            ax.set_ylabel("Token accuracy within bucket (%)")
            ax.set_title("Accuracy by sentence-length bucket")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures / "length_bucket_accuracy.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated.append("figures/length_bucket_accuracy.png")

    # Corpus-level figures, computed from the supplied training file.
    train_path = data_path / "train.tags"
    if train_path.exists():
        train = read_tagged(str(train_path))
        tag_counts: Counter[str] = Counter(tag for _, tags in train for tag in tags)
        top = tag_counts.most_common(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([item[0] for item in top], [item[1] for item in top])
        ax.set_ylabel("Training token count")
        ax.set_title("Most frequent POS tags in the training corpus")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(figures / "training_tag_distribution.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated.append("figures/training_tag_distribution.png")
        sent_lens = [len(words) for words, _ in train]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.hist(sent_lens, bins=30)
        ax.set_xlabel("Sentence length (tokens)")
        ax.set_ylabel("Number of sentences")
        ax.set_title("Training sentence-length distribution")
        fig.tight_layout()
        fig.savefig(figures / "sentence_length_distribution.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated.append("figures/sentence_length_distribution.png")

    return generated


def build_report_draft(output_dir: str | Path, cfg: dict[str, Any], best_variant: str | None = None) -> Path:
    """Create an editable Markdown report draft that uses only computed outputs."""
    root = Path(output_dir)
    summary_path = root / "experiment_summary.json"
    metrics_path = root / "dev_metrics.json"
    exp_path = root / "experiments.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    experiments: list[dict[str, str]] = []
    if exp_path.exists():
        with exp_path.open(encoding="utf-8", newline="") as handle:
            experiments = list(csv.DictReader(handle))
    experiments.sort(key=lambda row: float(row.get("val_accuracy_pct") or -1), reverse=True)
    if best_variant is None:
        best_variant = summary.get("best_variant")

    def p(v: Any, digits: int = 2) -> str:
        try:
            return f"{float(v) * 100:.{digits}f}%"
        except (TypeError, ValueError):
            return "not available"

    best_row = next((r for r in experiments if r.get("variant") == best_variant), {})
    smoke_only = summary.get("max_sentences_for_run") is not None
    status_line = (f"**SMOKE TEST ONLY - DO NOT SUBMIT OR REPORT THESE SCORES.** This draft used only {summary.get('max_sentences_for_run')} training sentences for pipeline verification." if smoke_only else "**Status:** Automatically generated results draft; verify attribution and add the official PA1 HMM result before submission.")
    lines = [
        "# Neural POS Tagging Under a 100-D Token Representation Budget",
        "",
        "**Group member 1:** [name and roll number]  ",
        "**Group member 2:** [name and roll number]  ",
        "**Course:** DAL-585, Programming Assignment 2  ",
        status_line,
        "",
        "## Abstract",
    ]
    if metrics:
        lines += [
            f"We implement a trainable neural POS tagger combining the selected lexical/character/shape inputs with a recurrent encoder and {cfg.get('decoder', 'softmax')} decoding. The selected experimental variant, **{best_variant or 'not selected'}**, achieved {p(float(best_row.get('val_accuracy_pct', 'nan')) / 100)} on the sentence-level validation split. On the provided development set, it achieved {p(metrics.get('token_accuracy'))} token accuracy across {metrics.get('token_count', 'N/A')} tokens, with {p(metrics.get('surface_oov_accuracy'))} accuracy on {metrics.get('surface_oov_count', 'N/A')} exact-surface OOV tokens. All results must be interpreted in light of the fixed split and search protocol.",
        ]
    else:
        lines += ["This section is populated after the experiment pipeline has completed. Do not insert unmeasured or copied performance values."]
    lines += [
        "",
        "## 1. Introduction and Related Work",
        "POS tagging assigns a Penn Treebank label to each token. Lexical identity helps disambiguate frequent words, while spelling and morphology can help with rare or unseen forms. Character-aware word representations have been studied for POS tagging by dos Santos and Zadrozny (2014), and Ma and Hovy (2016) combine character CNNs, a BiLSTM, and a CRF for sequence labeling. The current assignment prohibits pretrained word vectors, so all learned embeddings are trained from the supplied corpus.",
        "",
        "## 2. Data and Vocabulary",
        f"The supplied training set contains {summary.get('train_sentences_total', 'N/A')} sentences and {summary.get('train_tokens_total', 'N/A')} tokens before any test-only smoke subset is applied. Experiments reserve a deterministic sentence-level holdout from the training set for selection. The provided development labels are used only for final reporting and error analysis, not for hyperparameter selection.",
        "",
        ("Words are mapped through an exact-case, digit-normalized lexical vocabulary." if cfg.get("case_sensitive_lookup", False) else "Words are mapped through a lowercased, digit-normalized lexical vocabulary.") + " The character encoder receives original surface spelling, and a compact shape embedding captures capitalization, digit, punctuation, and hyphen patterns when enabled. Unknown lexical forms still receive character and shape features when those encoders are enabled. OOV is reported both as exact-surface OOV and normalized-lexicon OOV, because the two definitions are not equivalent.",
        "",
        "## 3. Model Architecture",
        f"For each token w_t, we concatenate its trainable word vector, character representation, and shape embedding: x_t = [e_w(w_t); e_c(w_t); e_s(w_t)]. The token representation is {int(cfg.get('word_dim', 0)) + int(cfg.get('shape_dim', 0)) + (int(cfg.get('char_out', 0)) if cfg.get('char_type') != 'none' else 0)} dimensional, respecting the PA2 limit of at most 100 dimensions. The sequence is encoded with a {cfg.get('layers')} layer {'bidirectional ' if cfg.get('bidirectional') else ''}LSTM and decoded by {cfg.get('decoder')}. See `figures/architecture.png`.",
        "",
        "## 4. Training Objective and Regularization",
        f"The model uses AdamW (learning rate {cfg.get('lr')}, weight decay {cfg.get('wd')}), batch size {cfg.get('bs')}, gradient clipping at {cfg.get('clip')}, and dropout probabilities {cfg.get('emb_drop')}/{cfg.get('lstm_drop')}/{cfg.get('out_drop')} for token input/recurrent/output layers respectively. Frequency-aware word dropout encourages the model to rely on character and context features for rare words. Early stopping monitors training-derived validation accuracy; the best validation checkpoint is used for experimental model selection.",
        "",
        "## 5. Results and Ablations",
        "| Variant | Validation accuracy | Surface-OOV accuracy | Lookup-OOV accuracy | Parameters | Epochs | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in experiments[:25]:
        lines.append(f"| {row.get('variant', '')} | {row.get('val_accuracy_pct', 'N/A')}% | {row.get('val_surface_oov_accuracy_pct', 'N/A')}% | {row.get('val_lookup_oov_accuracy_pct', 'N/A')}% | {row.get('params', 'N/A')} | {row.get('epochs_run', 'N/A')} | {row.get('seconds', 'N/A')} |")
    if metrics:
        lines += [
            "",
            "### Development set",
            f"- Token accuracy: **{p(metrics.get('token_accuracy'))}** ({metrics.get('correct_tokens', 'N/A')}/{metrics.get('token_count', 'N/A')} correct tokens).",
            f"- Exact sentence accuracy: **{p(metrics.get('exact_sentence_accuracy'))}** ({metrics.get('exact_sentence_count', 'N/A')}/{metrics.get('sentence_count', 'N/A')} fully correct sentences).",
            f"- Exact-surface OOV accuracy: **{p(metrics.get('surface_oov_accuracy'))}** on {metrics.get('surface_oov_count', 'N/A')} tokens.",
            f"- Lowercased/digit-normalized OOV accuracy: **{p(metrics.get('normalized_oov_accuracy'))}** on {metrics.get('normalized_oov_count', 'N/A')} tokens.",
            f"- Selected-model lexical-lookup OOV accuracy: **{p(metrics.get('lookup_oov_accuracy'))}** on {metrics.get('lookup_oov_count', 'N/A')} tokens.",
            "",
            "### Error analysis",
            "Use `per_tag_metrics.csv`, `confusion_pairs.csv`, `error_examples.csv`, and the plots in `figures/` to report the most frequent confusions and the tags with the highest error rates. Distinguish errors caused by grammatical ambiguity from those caused by unseen spelling patterns; this requires inspection of the saved token contexts rather than speculation.",
        ]
    lines += [
        "",
        "### PA1 HMM comparison (required by the assignment)",
        "Insert the group's own PA1 HMM result on this same provided development set, ideally by rerunning the exact PA1 evaluation script. The supplied peer report mentions HMM scores of 96.34% and 96.32%, but these have not been independently verified by this pipeline and should not be presented as results from the current implementation without confirmation.",
        "",
        "## 6. Runtime and Reproducibility",
        f"Python {summary.get('run_config', {}).get('python', 'N/A')}; PyTorch {summary.get('run_config', {}).get('torch', 'N/A')}; NumPy {summary.get('run_config', {}).get('numpy', 'N/A')}. Experiment device: {summary.get('run_config', {}).get('device', 'N/A')}.",
        "",
        "| Stage | Measured time | Details |",
        "|---|---:|---|",
        f"| Final full-data refit inside `run_all.py` | {summary.get('final_model_runtime', {}).get('final_fit_seconds', 'not recorded')} s | {summary.get('final_model_runtime', {}).get('final_fit_epochs', 'N/A')} epochs; {summary.get('final_model_runtime', {}).get('final_parameters', 'N/A')} parameters |",
        f"| `run_tagger.py` on the full development input | {summary.get('final_model_runtime', {}).get('run_tagger_wall_seconds', 'not recorded')} s | Includes model load and process startup; {summary.get('final_model_runtime', {}).get('dev_tokens', 'N/A')} tokens |",
        "| Individual ablation runtime | See `experiments.csv` | Per-variant values and epoch counts are recorded from the measured run. |",
        "",
        "## 7. Limitations",
        "Hyperparameter comparisons use a single fixed random seed unless replicated explicitly. Repeated experiments with additional seeds would better quantify variance. The provided dev set is a development benchmark rather than an unseen blind test set; therefore dev accuracy is not an estimate of blind-test performance. The official PA1 baseline remains to be verified.",
        "",
        "## 8. Conclusion",
        "The final conclusion should be written after examining the completed ablation table, validation curves, and development-set error analysis. It should identify which representation and regularization choices helped on the held-out training split, whether those improvements carried over to the provided development set, and which POS confusions remain difficult.",
        "",
        "## References",
        "Cícero Nogueira dos Santos and Bianca Zadrozny. 2014. Learning Character-level Representations for Part-of-Speech Tagging. *Proceedings of the 31st International Conference on Machine Learning*, 1818–1826.",
        "Xuezhe Ma and Eduard Hovy. 2016. End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1064–1074.",
        "Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1993. Building a Large Annotated Corpus of English: The Penn Treebank. *Computational Linguistics*, 19(2):313–330.",
        "",
        "## Figures to include",
        *[f"- `{path}`" for path in sorted(str(p.relative_to(root)).replace('\\\\', '/') for p in (root / 'figures').glob('*') if p.is_file())],
        "",
    ]
    report_path = root / "report_draft.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_acl_tex_report(output_dir: str | Path, cfg: dict[str, Any], best_variant: str | None = None) -> Path:
    """Generate an editable ACL-style LaTeX source and BibTeX file from measured results.

    Compilation requires the official ACL LaTeX style files (`acl.sty` and
    `acl_natbib.bst`), available from the ACL Anthology/Overleaf ACL template.
    The source is deliberately left editable so group details and PA1 baselines
    can be verified before hardcopy submission.
    """
    root = Path(output_dir)
    summary_path, metrics_path = root / "experiment_summary.json", root / "dev_metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    records: list[dict[str, str]] = []
    exp_path = root / "experiments.csv"
    if exp_path.exists():
        with exp_path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
    records.sort(key=lambda row: float(row.get("val_accuracy_pct") or -1), reverse=True)
    if best_variant is None:
        best_variant = summary.get("best_variant")
    best_record = next((record for record in records if record.get("variant") == best_variant), {})

    def latex_escape(value: Any) -> str:
        text = "" if value is None else str(value)
        return (text.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
                .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
                .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))

    def pct(value: Any) -> str:
        try:
            if value is None or value == "":
                return "--"
            return f"{float(value) * 100:.2f}\\%"
        except (TypeError, ValueError):
            return "--"

    train_sentences = summary.get("train_sentences_total", "N/A")
    train_tokens = summary.get("train_tokens_total", "N/A")
    val_sentences = summary.get("validation_sentences", "N/A")
    val_tokens = summary.get("validation_tokens", "N/A")
    dev_tokens = metrics.get("token_count", "N/A")
    dev_sents = metrics.get("sentence_count", "N/A")
    best_val = best_record.get("val_accuracy_pct")
    decoder_name = "linear-chain CRF" if cfg.get("decoder") == "crf" else "token-level softmax"
    biword = "bidirectional LSTM" if cfg.get("bidirectional", True) else "unidirectional LSTM"
    char_dim = cfg.get("char_out", 0) if cfg.get("char_type") != "none" else 0
    input_dim = int(cfg.get("word_dim", 0)) + int(cfg.get("shape_dim", 0)) + int(char_dim)
    abstract = (
        f"We study neural part-of-speech tagging under the assignment's constraint that the concatenated token input representation is at most 100 dimensions and no pretrained word vectors are allowed. "
        f"Our selected variant is {best_variant or 'not yet selected'}, using {cfg.get('word_dim')}-dimensional trainable word embeddings, {cfg.get('char_type')} character features ({char_dim} dimensions), and {cfg.get('shape_dim')}-dimensional token-shape embeddings. "
    )
    if metrics:
        abstract += f"It achieves {pct(metrics.get('token_accuracy'))} token accuracy on the provided development set and {pct(metrics.get('surface_oov_accuracy'))} accuracy on exact-surface OOV tokens. "
    else:
        abstract += "Development-set results will be filled in when the labeled evaluation is run. "
    abstract += "A fixed sentence-level validation split from the training corpus is used for model selection, and ablations isolate character encoding, recurrence direction, decoding, representation allocation, model capacity, dropout, and optimization choices."
    if summary.get("max_sentences_for_run") is not None:
        abstract = "SMOKE TEST ONLY - NOT FOR REPORTING. " + abstract

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{acl}",
        r"\usepackage{times}",
        r"\usepackage{latexsym}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{multirow}",
        r"\usepackage{amsmath}",
        r"\usepackage{url}",
        r"\title{Neural POS Tagging Under a 100-Dimensional Input Budget}",
        r"\author{Group Member 1 \\ Roll Number \\ \And Group Member 2 \\ Roll Number}",
        r"\begin{document}",
        r"\maketitle",
        *([r"\begin{center}\textbf{SMOKE TEST ONLY - DO NOT SUBMIT OR REPORT THESE SCORES.}\end{center}"] if summary.get("max_sentences_for_run") is not None else []),
        r"\begin{abstract}", latex_escape(abstract), r"\end{abstract}",
        r"\section{Introduction}",
        r"Part-of-speech (POS) tagging assigns a syntactic label to each token in a sentence. Many tokens are locally ambiguous: context may be needed to distinguish a noun from an adjective, a verb form from a participle, or a preposition from a particle. This project implements a neural sequence tagger for the Penn Treebank tag inventory using only trainable representations learned from the provided training data. The implementation obeys PA2's no-pretrained-vector rule and limits the concatenated per-token input representation to 100 dimensions.",
        r"\section{Related Work}",
        r"Character information complements lexical embeddings by allowing a model to learn spelling and morphology patterns shared across word forms. Dos Santos and Zadrozny (\citeyear{santos2014}) show the utility of character-level representations for POS tagging. Ma and Hovy (\citeyear{mahovy2016}) combine character CNNs, a BiLSTM, and a CRF for sequence labeling. We use these ideas as motivation, but all vectors and parameters are initialized and trained using the supplied corpus; no external pretrained embeddings are used.",
        r"\section{Data and Vocabulary}",
        f"The supplied training corpus contains {train_sentences} sentences and {train_tokens} tokens. Model selection uses a deterministic sentence-level training/validation split; the validation partition contains {val_sentences} sentences and {val_tokens} tokens. The separate development set contains {dev_sents} sentences and {dev_tokens} tokens. Its gold tags are not used to choose hyperparameters.",
        (r"Lexical lookup maps each token to an exact-case, digit-normalized form." if cfg.get("case_sensitive_lookup", False) else r"Lexical lookup maps each token to a lowercased, digit-normalized form.") + r" The character encoder sees original surface spelling, including capitalization and punctuation, while an optional shape embedding represents broad orthographic patterns. A word absent from the selected model vocabulary maps to the unknown-word ID; enabled character and shape features remain available. We report exact-surface OOV, lowercased/digit-normalized OOV, and OOV under the selected model lookup separately.",
        r"\section{Model Architecture}",
        f"For token $w_t$, the input representation is $x_t=[e_w(\\operatorname{{norm}}(w_t)); e_c(w_t); e_s(w_t)] \\in \\mathbb{{R}}^{{{input_dim}}}$, where the components are a {cfg.get('word_dim')}-D trainable word embedding, a {char_dim}-D {cfg.get('char_type')} character representation, and a {cfg.get('shape_dim')}-D shape embedding. Thus, the input representation uses {input_dim} dimensions and satisfies the PA2 limit. The sequence is processed by a {cfg.get('layers')}-layer {biword} with {cfg.get('hidden')} hidden units per direction when bidirectional; dropout is applied to the input and output representations. The decoder is {decoder_name}. Figure~\\ref{{fig:arch}} summarizes the pipeline.",
        r"\begin{figure*}[t]", r"\centering", r"\includegraphics[width=0.96\textwidth]{figures/architecture.png}", r"\caption{Character-aware neural POS tagger. The token representation is constrained to at most 100 dimensions.}", r"\label{fig:arch}", r"\end{figure*}",
        r"\section{Training Objective and Model Selection}",
        f"The selected configuration uses AdamW with learning rate {cfg.get('lr')}, weight decay {cfg.get('wd')}, batch size {cfg.get('bs')}, gradient clipping at {cfg.get('clip')}, and input/recurrent/output dropout of {cfg.get('emb_drop')}/{cfg.get('lstm_drop')}/{cfg.get('out_drop')}. Frequency-aware word dropout replaces some lexical IDs with the unknown ID during training. For a softmax decoder, the objective is token-level cross-entropy; for a CRF decoder, the objective is the negative log-likelihood of the gold tag sequence. Early stopping monitors validation accuracy, with validation loss used to break near-ties and to adjust the learning rate.",
        r"\section{Experiments and Results}",
        "The ablation suite tests word-only features, word plus shape, character CNN versus character LSTM, unidirectional versus bidirectional context, softmax versus CRF decoding, alternative word/character/shape dimension allocations, recurrent capacity and depth, dropout, word dropout, learning rate, batch size, and label smoothing. Table~\\ref{tab:ablations} records the measured outcomes on the same internal validation split.",
        r"\begin{table*}[t]", r"\centering", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Variant & Val. acc. & Surface OOV & Lookup OOV & Params & Epochs & Time (s) " + r"\\", r"\midrule",
    ]
    for record in records:
        variant = latex_escape(record.get("variant", ""))
        val_acc = latex_escape(record.get("val_accuracy_pct", "--")) + r"\%" if record.get("val_accuracy_pct") not in (None, "") else "--"
        oov = latex_escape(record.get("val_surface_oov_accuracy_pct", "--")) + r"\%" if record.get("val_surface_oov_accuracy_pct") not in (None, "") and record.get("val_surface_oov_accuracy_pct") != "nan" else "--"
        lookup_oov = latex_escape(record.get("val_lookup_oov_accuracy_pct", "--")) + r"\%" if record.get("val_lookup_oov_accuracy_pct") not in (None, "") and record.get("val_lookup_oov_accuracy_pct") != "nan" else "--"
        params = record.get("params", "--")
        epochs = record.get("epochs_run", "--")
        seconds = record.get("seconds", "--")
        lines.append(f"{variant} & {val_acc} & {oov} & {lookup_oov} & {params} & {epochs} & {seconds} " + chr(92) * 2)
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\caption{Ablations on the fixed training-derived validation split. The complete metrics and run metadata are also provided as CSV/JSON artifacts.}", r"\label{tab:ablations}", r"\end{table*}"]
    if metrics:
        lines += [
            f"On the development set, the selected model tags {metrics.get('correct_tokens', 'N/A')} of {dev_tokens} tokens correctly ({pct(metrics.get('token_accuracy'))}). Exact-sentence accuracy is {pct(metrics.get('exact_sentence_accuracy'))}. Exact-surface OOV accuracy is {pct(metrics.get('surface_oov_accuracy'))} over {metrics.get('surface_oov_count', 'N/A')} tokens; lowercased/digit-normalized OOV accuracy is {pct(metrics.get('normalized_oov_accuracy'))} over {metrics.get('normalized_oov_count', 'N/A')} tokens; model-lookup OOV accuracy is {pct(metrics.get('lookup_oov_accuracy'))} over {metrics.get('lookup_oov_count', 'N/A')} tokens.",
            r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\linewidth]{figures/experiment_accuracy.png}", r"\caption{Ablation and hyperparameter comparison on the training-derived validation split.}", r"\label{fig:ablation}", r"\end{figure}",
        ]
        if (root / "figures" / "learning_curves.png").exists():
            lines += [r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\linewidth]{figures/learning_curves.png}", r"\caption{Validation accuracy on the training-derived holdout over epochs.}", r"\label{fig:learning}", r"\end{figure}"]
        if (root / "figures" / "tag_confusions.png").exists():
            lines += [r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\linewidth]{figures/tag_confusions.png}", r"\caption{Frequent directed tag confusions on the provided development set.}", r"\label{fig:confusions}", r"\end{figure}"]
    lines += [
        r"\paragraph{PA1 HMM baseline.} The assignment asks for a comparison with the group's PA1 HMM result. Insert the verified PA1 accuracy measured on the same development files here. The peer report included with the initial materials mentions 96.34\% and 96.32\%, but this pipeline has not independently reproduced those figures; they should not be attributed to the present implementation without verification.",
        r"\section{Error Analysis}",
        r"The pipeline exports per-tag recall/accuracy, directed confusion counts, token-level predictions, and contextual examples in CSV files. Use these tables to identify which POS labels contribute most errors, distinguishing high error counts caused by frequent labels from high error rates caused by difficult labels. The figure and tables must be interpreted from the actual run; do not infer error causes from counts alone.",
        r"\section{Runtime and Reproducibility}",
        f"The selected configuration uses {cfg.get('word_dim')} word, {char_dim} character, and {cfg.get('shape_dim')} shape dimensions. All experiments share a fixed sentence-level split and random seed {summary.get('run_config', {}).get('seed', 42)}; a single seed does not measure run-to-run variance.",
        r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrr}", r"\toprule", r"Stage & Seconds & Details " + r"\\", r"\midrule",
        f"Full-data refit (run\\_all.py) & {summary.get('final_model_runtime', {}).get('final_fit_seconds', '--')} & {summary.get('final_model_runtime', {}).get('final_fit_epochs', '--')} epochs " + chr(92) * 2,
        f"run\\_tagger.py on dev & {summary.get('final_model_runtime', {}).get('run_tagger_wall_seconds', '--')} & {summary.get('final_model_runtime', {}).get('dev_tokens', '--')} tokens " + chr(92) * 2,
        r"\bottomrule", r"\end{tabular}", r"\caption{Measured deployment refit and inference runtime. Per-variant training times and parameter counts are provided in the experiment CSV.}", r"\label{tab:runtime}", r"\end{table}",
        r"\section{Limitations}",
        r"Repeated comparisons on one validation split can overfit the validation criterion. The provided development set is used for final diagnostics, so its score should not be interpreted as a blind-test score. Future checks should rerun the best one or two configurations with multiple seeds and verify the required PA1 baseline using the original PA1 evaluation code.",
        r"\section{Conclusion}",
        f"The experiment suite evaluates a trainable, no-pretrained-vector POS tagger under a {input_dim}-D token-input budget. The final conclusions should be based on the observed ablation rankings, validation curves, and development errors, rather than assuming that a more complex architecture is necessarily better.",
        r"\bibliographystyle{acl_natbib}", r"\bibliography{references}",
        r"\end{document}",
    ]
    tex_path = root / "report_draft.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bib = r"""@inproceedings{santos2014,
  author = {dos Santos, C{\'i}cero Nogueira and Zadrozny, Bianca},
  title = {Learning Character-level Representations for Part-of-Speech Tagging},
  booktitle = {Proceedings of the 31st International Conference on Machine Learning},
  pages = {1818--1826},
  year = {2014},
  url = {https://proceedings.mlr.press/v32/santos14.html}
}

@inproceedings{mahovy2016,
  author = {Ma, Xuezhe and Hovy, Eduard},
  title = {End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1064--1074},
  year = {2016},
  url = {https://aclanthology.org/P16-1101/}
}

@article{marcus1993,
  author = {Marcus, Mitchell P. and Santorini, Beatrice and Marcinkiewicz, Mary Ann},
  title = {Building a Large Annotated Corpus of English: The Penn Treebank},
  journal = {Computational Linguistics},
  volume = {19},
  number = {2},
  pages = {313--330},
  year = {1993},
  url = {https://aclanthology.org/J93-2004/}
}
"""
    (root / "references.bib").write_text(bib, encoding="utf-8")
    return tex_path
