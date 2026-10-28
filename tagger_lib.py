"""Shared implementation for a character-aware BiLSTM POS tagger.

The model uses no pretrained vectors. The concatenated per-token input
representation is constrained to at most 100 dimensions, as required by PA2.
"""
from __future__ import annotations

import copy
import math
import random
import re
import time
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK = 0, 1
MODEL_FILE = "tagger_model.pt"
N_SHAPES = 8

DEFAULT_CFG: dict[str, Any] = {
    "word_dim": 50,
    "char_out": 40,
    "shape_dim": 10,
    "char_emb": 25,
    "char_type": "cnn",        # cnn | lstm | none
    "max_chars": 24,
    "hidden": 256,              # units per recurrent direction
    "layers": 2,
    "bidirectional": True,
    "decoder": "softmax",       # softmax | crf
    "case_sensitive_lookup": False, # preserve case in lexical keys when True; digit normalization remains on
    "emb_drop": 0.35,
    "lstm_drop": 0.25,
    "out_drop": 0.35,
    "wdrop_alpha": 1.0,         # word-drop probability = alpha / (alpha + count)
    "lr": 2e-3,
    "wd": 1e-5,
    "bs": 96,
    "epochs": 18,
    "patience": 4,
    "min_delta": 0.00003,
    "clip": 5.0,
    "label_smooth": 0.0,
}


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail early on invalid model settings and the assignment's 100-D limit."""
    required = set(DEFAULT_CFG)
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Config missing fields: {sorted(missing)}")
    if cfg["char_type"] not in {"cnn", "lstm", "none"}:
        raise ValueError("char_type must be one of: cnn, lstm, none")
    if cfg["decoder"] not in {"softmax", "crf"}:
        raise ValueError("decoder must be softmax or crf")
    if not isinstance(cfg["case_sensitive_lookup"], bool):
        raise ValueError("case_sensitive_lookup must be a boolean")
    if int(cfg["word_dim"]) < 0 or int(cfg["shape_dim"]) < 0:
        raise ValueError("word_dim and shape_dim must be non-negative")
    if cfg["char_type"] != "none" and int(cfg["char_out"]) <= 0:
        raise ValueError("char_out must be positive when a character encoder is used")
    vec_dim = int(cfg["word_dim"]) + int(cfg["shape_dim"])
    if cfg["char_type"] != "none":
        vec_dim += int(cfg["char_out"])
    if not 0 < vec_dim <= 100:
        raise ValueError(f"Input representation is {vec_dim}D; it must be in [1, 100]")
    if cfg["word_dim"] == 0 and cfg["shape_dim"] == 0 and cfg["char_type"] == "none":
        raise ValueError("At least one token representation must be enabled")
    if int(cfg["hidden"]) < 1 or int(cfg["layers"]) < 1:
        raise ValueError("hidden and layers must be positive")
    if cfg["char_type"] == "lstm" and int(cfg["char_out"]) % 2:
        raise ValueError("char_out must be even for the bidirectional character LSTM")
    if int(cfg["max_chars"]) < 8:
        raise ValueError("max_chars must be at least 8")
    for key in ("emb_drop", "lstm_drop", "out_drop", "label_smooth"):
        if not 0 <= float(cfg[key]) < 1:
            raise ValueError(f"{key} must be in [0, 1)")
    if float(cfg["lr"]) <= 0 or int(cfg["bs"]) <= 0 or int(cfg["epochs"]) <= 0:
        raise ValueError("lr, bs and epochs must be positive")
    if float(cfg["wd"]) < 0 or float(cfg["wdrop_alpha"]) < 0:
        raise ValueError("wd and wdrop_alpha must be non-negative")
    if float(cfg["clip"]) <= 0 or int(cfg["patience"]) < 1 or float(cfg["min_delta"]) < 0:
        raise ValueError("clip must be positive, patience >= 1, and min_delta non-negative")


def norm(word: str) -> str:
    """Lower-case and normalize digits, retaining lexical sharing across numbers."""
    return re.sub(r"\d", "0", word.lower())


def lexical_key(word: str, case_sensitive_lookup: bool = False) -> str:
    """Create the lexical lookup key; optionally preserve case while normalizing digits."""
    source = word if case_sensitive_lookup else word.lower()
    return re.sub(r"\d", "0", source)


def shape_id(word: str) -> int:
    """Map a surface token to a compact capitalization/punctuation signature."""
    if not any(ch.isalnum() for ch in word):
        return 5
    if any(ch.isdigit() for ch in word):
        return 3
    letters = [ch for ch in word if ch.isalpha()]
    has_hyphen = "-" in word
    if len(letters) > 1 and all(ch.isupper() for ch in letters):
        return 2
    if word[0].isupper():
        return 7 if has_hyphen else 1
    if letters and all(ch.islower() for ch in letters):
        return 6 if has_hyphen else 0
    return 4


def read_tagged(path: str) -> list[tuple[list[str], list[str]]]:
    """Read one whitespace-tokenized `word/TAG` sentence per non-empty line."""
    sentences: list[tuple[list[str], list[str]]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            tokens = line.split()
            if not tokens:
                continue
            words: list[str] = []
            tags: list[str] = []
            for token in tokens:
                if "/" not in token:
                    raise ValueError(f"Malformed token at {path}:{line_no}: {token!r}")
                word, tag = token.rsplit("/", 1)
                if not word or not tag:
                    raise ValueError(f"Malformed token at {path}:{line_no}: {token!r}")
                words.append(word)
                tags.append(tag)
            if len(words) != len(tags):
                raise AssertionError("Internal parse error")
            sentences.append((words, tags))
    if not sentences:
        raise ValueError(f"No non-empty tagged sentences found in {path}")
    return sentences


def read_sents(path: str) -> list[list[str]]:
    """Read tokenized sentences, preserving blank lines for output alignment."""
    with open(path, encoding="utf-8") as handle:
        return [line.split() for line in handle.read().splitlines()]


def split_train_val(
    sentences: Sequence[tuple[list[str], list[str]]], val_fraction: float = 0.05, seed: int = 42
) -> tuple[list[tuple[list[str], list[str]]], list[tuple[list[str], list[str]]]]:
    """Deterministically split by sentence, not by token, to avoid token leakage."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    idx = list(range(len(sentences)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, min(len(sentences) - 1, round(len(sentences) * val_fraction)))
    val_idx = set(idx[:n_val])
    train = [sentences[i] for i in range(len(sentences)) if i not in val_idx]
    val = [sentences[i] for i in range(len(sentences)) if i in val_idx]
    return train, val


class Vocab:
    """Training-derived vocabularies and feature encodings.

    `surface_words` is retained separately from normalized lookup so that the
    report's strict OOV metric (exact surface form absent from training) can
    differ from normalized-lexicon OOV.
    """

    def __init__(
        self,
        training_sentences: Sequence[tuple[list[str], list[str]]],
        tag_inventory: Iterable[str] | None = None,
        case_sensitive_lookup: bool = False,
    ) -> None:
        self.case_sensitive_lookup = bool(case_sensitive_lookup)
        word_counts: Counter[str] = Counter()
        char_counts: Counter[str] = Counter()
        surface_words: set[str] = set()
        observed_tags: set[str] = set()
        for words, tags in training_sentences:
            for word, tag in zip(words, tags):
                word_counts[lexical_key(word, self.case_sensitive_lookup)] += 1
                surface_words.add(word)
                observed_tags.add(tag)
                char_counts.update(word)
        self.word2id: dict[str, int] = {"<pad>": PAD, "<unk>": UNK}
        for word in sorted(word_counts):
            self.word2id[word] = len(self.word2id)
        self.counts = np.zeros(len(self.word2id), dtype=np.float32)
        for word, count in word_counts.items():
            self.counts[self.word2id[word]] = count
        # Reserved ids: 0 padding, 1 unknown, 2 start, 3 end.
        self.char2id: dict[str, int] = {"\x00": 0, "\x01": 1, "\x02": 2, "\x03": 3}
        for char in sorted(char_counts):
            if char_counts[char] >= 2 and char not in self.char2id:
                self.char2id[char] = len(self.char2id)
        all_tags = sorted(set(tag_inventory) if tag_inventory is not None else observed_tags)
        if not all_tags:
            raise ValueError("No POS tags found")
        self.tags = all_tags
        self.tag2id = {tag: index for index, tag in enumerate(self.tags)}
        missing = observed_tags - set(self.tag2id)
        if missing:
            raise ValueError(f"tag_inventory omits observed tags: {sorted(missing)}")
        self.surface_words = surface_words
        self._char_cache: dict[tuple[str, int], np.ndarray] = {}

    def chars_of(self, word: str, max_chars: int) -> np.ndarray:
        key = (word, max_chars)
        cached = self._char_cache.get(key)
        if cached is not None:
            return cached
        # Include boundary markers and preserve both edges for rare/long words.
        if len(word) > max_chars - 2:
            keep_suffix = max_chars - 7
            clipped = word[:5] + word[-keep_suffix:]
        else:
            clipped = word
        char_ids = [2] + [self.char2id.get(ch, 1) for ch in clipped] + [3]
        array = np.zeros(max_chars, dtype=np.int64)
        array[: min(max_chars, len(char_ids))] = char_ids[:max_chars]
        self._char_cache[key] = array
        return array

    def encode(self, words: Sequence[str], tags: Sequence[str] | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {
            "w": list(words),
            "wid": np.asarray([self.word2id.get(lexical_key(word, self.case_sensitive_lookup), UNK) for word in words], dtype=np.int64),
            "sid": np.asarray([shape_id(word) for word in words], dtype=np.int64),
            "surface_oov": np.asarray([word not in self.surface_words for word in words], dtype=np.bool_),
        }
        if tags is not None:
            unknown = sorted(set(tags) - set(self.tag2id))
            if unknown:
                raise ValueError(f"Unknown gold tags encountered: {unknown}")
            item["y"] = np.asarray([self.tag2id[tag] for tag in tags], dtype=np.int64)
        return item

    def state(self) -> dict[str, Any]:
        return {
            "word2id": self.word2id,
            "char2id": self.char2id,
            "tags": self.tags,
            "counts": self.counts,
            "surface_words": self.surface_words,
            "case_sensitive_lookup": self.case_sensitive_lookup,
        }

    @staticmethod
    def from_state(state: dict[str, Any]) -> "Vocab":
        vocab = Vocab.__new__(Vocab)
        vocab.word2id = state["word2id"]
        vocab.char2id = state["char2id"]
        vocab.tags = state["tags"]
        vocab.counts = np.asarray(state["counts"], dtype=np.float32)
        vocab.tag2id = {tag: index for index, tag in enumerate(vocab.tags)}
        vocab.surface_words = set(state.get("surface_words", vocab.word2id.keys()))
        vocab.case_sensitive_lookup = bool(state.get("case_sensitive_lookup", False))
        vocab._char_cache = {}
        return vocab


def collate(
    items: Sequence[dict[str, Any]], vocab: Vocab, cfg: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    if any(len(item["w"]) == 0 for item in items):
        raise ValueError("Empty sentences must be filtered before batching")
    batch_size = len(items)
    max_tokens = max(len(item["w"]) for item in items)
    wid = np.zeros((batch_size, max_tokens), dtype=np.int64)
    sid = np.zeros((batch_size, max_tokens), dtype=np.int64)
    labels = np.full((batch_size, max_tokens), -100, dtype=np.int64)
    token_lengths = np.zeros(batch_size, dtype=np.int64)
    # Encode each distinct surface word only once per batch.
    unique: dict[str, int] = {"": 0}
    unique_words: list[str] = [""]
    token_unique_ids = np.zeros((batch_size, max_tokens), dtype=np.int64)
    for row, item in enumerate(items):
        length = len(item["w"])
        token_lengths[row] = length
        wid[row, :length] = item["wid"]
        sid[row, :length] = item["sid"]
        if "y" in item:
            labels[row, :length] = item["y"]
        for col, word in enumerate(item["w"]):
            unique_id = unique.get(word)
            if unique_id is None:
                unique_id = len(unique_words)
                unique[word] = unique_id
                unique_words.append(word)
            token_unique_ids[row, col] = unique_id
    char_array = np.stack([vocab.chars_of(word, int(cfg["max_chars"])) for word in unique_words])

    def to_device(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device)

    return (
        to_device(wid),
        to_device(sid),
        to_device(char_array),
        to_device(token_unique_ids),
        torch.from_numpy(token_lengths),  # pack_padded_sequence expects CPU lengths
        to_device(labels),
    )


class LinearChainCRF(nn.Module):
    """Small batched linear-chain CRF with Viterbi decoding."""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))  # previous -> current
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, time_steps, _ = emissions.shape
        batch_index = torch.arange(batch, device=emissions.device)
        # Padded labels use -100 for cross-entropy; clamp them before indexing and
        # mask their contribution out below, so variable-length CRF batches are safe.
        safe_tags = tags.clamp_min(0)
        score = self.start_transitions[safe_tags[:, 0]] + emissions[batch_index, 0, safe_tags[:, 0]]
        for t in range(1, time_steps):
            active = lengths > t
            transition_score = self.transitions[safe_tags[:, t - 1], safe_tags[:, t]]
            emission_score = emissions[batch_index, t, safe_tags[:, t]]
            score = score + (transition_score + emission_score) * active.to(emissions.dtype)
        last_tags = safe_tags[batch_index, lengths.to(emissions.device) - 1]
        return score + self.end_transitions[last_tags]

    def _partition(self, emissions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, time_steps, _ = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, time_steps):
            candidates = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_alpha = torch.logsumexp(candidates, dim=1)
            alpha = torch.where((lengths > t).to(emissions.device).unsqueeze(1), next_alpha, alpha)
        return torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_device = lengths.to(emissions.device)
        return (self._partition(emissions, lengths_device) - self._gold_score(emissions, tags, lengths_device)).sum() / lengths_device.sum().clamp_min(1)

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        batch, time_steps, _ = emissions.shape
        lengths_device = lengths.to(emissions.device)
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        histories: list[torch.Tensor] = []
        for t in range(1, time_steps):
            candidates = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            best_score, best_previous = candidates.max(dim=1)
            active = (lengths_device > t).unsqueeze(1)
            score = torch.where(active, best_score, score)
            histories.append(best_previous)
        score = score + self.end_transitions.unsqueeze(0)
        last = score.argmax(dim=1)
        results: list[list[int]] = []
        for row in range(batch):
            length = int(lengths[row])
            tag = int(last[row])
            path = [tag]
            for t in range(length - 1, 0, -1):
                tag = int(histories[t - 1][row, tag])
                path.append(tag)
            path.reverse()
            results.append(path)
        return results


class Tagger(nn.Module):
    def __init__(self, cfg: dict[str, Any], n_words: int, n_chars: int, n_tags: int) -> None:
        super().__init__()
        validate_config(cfg)
        self.cfg = copy.deepcopy(cfg)
        dimensions = 0
        self.wemb: nn.Embedding | None = None
        self.semb: nn.Embedding | None = None
        self.cemb: nn.Embedding | None = None
        if cfg["word_dim"] > 0:
            self.wemb = nn.Embedding(n_words, cfg["word_dim"], padding_idx=PAD)
            dimensions += cfg["word_dim"]
        if cfg["shape_dim"] > 0:
            self.semb = nn.Embedding(N_SHAPES, cfg["shape_dim"])
            dimensions += cfg["shape_dim"]
        if cfg["char_type"] != "none":
            self.cemb = nn.Embedding(n_chars, cfg["char_emb"], padding_idx=0)
            if cfg["char_type"] == "cnn":
                # Multi-width character convolutions (3 and 5) capture short affixes
                # as well as slightly longer orthographic/morphological patterns.
                c3 = int(cfg["char_out"]) // 2
                c5 = int(cfg["char_out"]) - c3
                self.conv3 = nn.Conv1d(cfg["char_emb"], c3, kernel_size=3, padding=1)
                self.conv5 = nn.Conv1d(cfg["char_emb"], c5, kernel_size=5, padding=2)
            else:
                self.clstm = nn.LSTM(
                    cfg["char_emb"], cfg["char_out"] // 2, batch_first=True, bidirectional=True
                )
            dimensions += cfg["char_out"]
        self.input_dim = dimensions
        directions = 2 if cfg["bidirectional"] else 1
        self.lstm = nn.LSTM(
            dimensions,
            cfg["hidden"],
            cfg["layers"],
            batch_first=True,
            bidirectional=cfg["bidirectional"],
            dropout=cfg["lstm_drop"] if cfg["layers"] > 1 else 0.0,
        )
        self.drop_in = nn.Dropout(cfg["emb_drop"])
        self.drop_out = nn.Dropout(cfg["out_drop"])
        self.fc = nn.Linear(directions * cfg["hidden"], n_tags)
        self.crf = LinearChainCRF(n_tags) if cfg["decoder"] == "crf" else None

    def char_encode(self, unique_chars: torch.Tensor) -> torch.Tensor:
        if self.cemb is None:
            raise RuntimeError("Character encoding requested but character features are disabled")
        embedded = self.cemb(unique_chars)  # unique words x max chars x char embedding
        mask = unique_chars != 0
        if self.cfg["char_type"] == "cnn":
            channels_first = embedded.transpose(1, 2)
            h3 = F.relu(self.conv3(channels_first))
            h5 = F.relu(self.conv5(channels_first))
            h3 = h3.masked_fill(~mask.unsqueeze(1), 0.0)
            h5 = h5.masked_fill(~mask.unsqueeze(1), 0.0)
            # Max-pool only over positions occupied by the word/boundary markers.
            return torch.cat((h3.max(dim=2).values, h5.max(dim=2).values), dim=-1)
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.clstm(packed)
        return torch.cat((hidden[0], hidden[1]), dim=-1)

    def forward(
        self,
        word_ids: torch.Tensor,
        shape_ids: torch.Tensor,
        unique_chars: torch.Tensor,
        unique_word_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        if self.wemb is not None:
            features.append(self.wemb(word_ids))
        if self.semb is not None:
            features.append(self.semb(shape_ids))
        if self.cemb is not None:
            features.append(self.char_encode(unique_chars)[unique_word_ids])
        token_vectors = self.drop_in(torch.cat(features, dim=-1))
        packed = pack_padded_sequence(token_vectors, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=word_ids.size(1))
        return self.fc(self.drop_out(output))

    def sequence_loss(self, emissions: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.crf is not None:
            return self.crf.neg_log_likelihood(emissions, labels, lengths)
        return F.cross_entropy(
            emissions.reshape(-1, emissions.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            label_smoothing=float(self.cfg["label_smooth"]),
        )

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
        if self.crf is not None:
            return self.crf.decode(emissions, lengths)
        predictions = emissions.argmax(dim=-1)
        return [predictions[row, : int(length)].tolist() for row, length in enumerate(lengths)]


def n_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep CUDA fast by default. Exact bitwise reproducibility across hardware/versions is not guaranteed.


@torch.no_grad()
def evaluate_model(
    model: Tagger,
    items: Sequence[dict[str, Any]],
    vocab: Vocab,
    cfg: dict[str, Any],
    device: torch.device,
    batch_size: int = 128,
    include_predictions: bool = False,
) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot evaluate an empty item list")
    model.eval()
    order = sorted(range(len(items)), key=lambda index: len(items[index]["w"]))
    correct = total = surface_oov_correct = surface_oov_total = 0
    lookup_oov_correct = lookup_oov_total = exact_sentences = 0
    loss_sum = token_count = 0.0
    preds: list[list[int] | None] = [None] * len(items)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = [items[index] for index in indices]
        word_ids, shape_ids, unique_chars, unique_ids, lengths, labels = collate(batch, vocab, cfg, device)
        emissions = model(word_ids, shape_ids, unique_chars, unique_ids, lengths)
        if all("y" in item for item in batch):
            loss = model.sequence_loss(emissions, labels, lengths)
            token_count_batch = int(lengths.sum())
            loss_sum += float(loss.item()) * token_count_batch
            token_count += token_count_batch
            decoded = model.decode(emissions, lengths)
        else:
            decoded = model.decode(emissions, lengths)
        for row, (index, item, predicted) in enumerate(zip(indices, batch, decoded)):
            preds[index] = predicted
            if "y" not in item:
                continue
            gold = item["y"]
            pred_arr = np.asarray(predicted, dtype=np.int64)
            hits = pred_arr == gold
            correct += int(hits.sum())
            total += len(gold)
            if bool(hits.all()):
                exact_sentences += 1
            surface_mask = item["surface_oov"]
            lookup_mask = item["wid"] == UNK
            surface_oov_total += int(surface_mask.sum())
            surface_oov_correct += int((hits & surface_mask).sum())
            lookup_oov_total += int(lookup_mask.sum())
            lookup_oov_correct += int((hits & lookup_mask).sum())
    result: dict[str, Any] = {
        "accuracy": correct / total if total else float("nan"),
        "tokens": total,
        "sentences": len(items),
        "exact_sentence_accuracy": exact_sentences / len(items) if total else float("nan"),
        "surface_oov_accuracy": surface_oov_correct / surface_oov_total if surface_oov_total else float("nan"),
        "surface_oov_tokens": surface_oov_total,
        "lookup_oov_accuracy": lookup_oov_correct / lookup_oov_total if lookup_oov_total else float("nan"),
        "lookup_oov_tokens": lookup_oov_total,
        "token_loss": loss_sum / token_count if token_count else float("nan"),
    }
    if include_predictions:
        result["predictions"] = [p if p is not None else [] for p in preds]
    return result


def _train_batches(
    model: Tagger,
    optimizer: torch.optim.Optimizer,
    train_items: Sequence[dict[str, Any]],
    vocab: Vocab,
    cfg: dict[str, Any],
    device: torch.device,
    rng: random.Random,
) -> float:
    n = len(train_items)
    # Length-aware batching greatly reduces padding; shuffle batch order each epoch.
    order = sorted(range(n), key=lambda i: len(train_items[i]["w"]) + rng.random() * 4.0)
    batches = [order[pos : pos + int(cfg["bs"])] for pos in range(0, n, int(cfg["bs"]))]
    rng.shuffle(batches)
    model.train()
    total_loss = 0.0
    total_tokens = 0
    word_drop_alpha = float(cfg["wdrop_alpha"])
    if word_drop_alpha > 0:
        pdrop = torch.from_numpy(
            word_drop_alpha / (word_drop_alpha + np.maximum(vocab.counts, 1.0))
        ).to(device)
        pdrop[:2] = 0.0
    else:
        pdrop = None
    for batch_indices in batches:
        batch = [train_items[index] for index in batch_indices]
        word_ids, shape_ids, unique_chars, unique_ids, lengths, labels = collate(batch, vocab, cfg, device)
        if pdrop is not None:
            replace = torch.rand(word_ids.shape, device=device) < pdrop[word_ids]
            word_ids = torch.where(replace, torch.full_like(word_ids, UNK), word_ids)
        logits = model(word_ids, shape_ids, unique_chars, unique_ids, lengths)
        loss = model.sequence_loss(logits, labels, lengths)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip"]))
        optimizer.step()
        n_tokens = int(lengths.sum())
        total_loss += float(loss.item()) * n_tokens
        total_tokens += n_tokens
    return total_loss / max(1, total_tokens)


def train_model(
    cfg: dict[str, Any],
    train_items: Sequence[dict[str, Any]],
    vocab: Vocab,
    seed: int,
    device: torch.device,
    val_items: Sequence[dict[str, Any]] | None = None,
    max_seconds: float | None = None,
    log=print,
) -> tuple[Tagger, dict[str, Any]]:
    """Train a tagger, selecting the best validation-accuracy checkpoint if supplied.

    The time guard stops cleanly between epochs. Early stopping monitors validation
    accuracy while ReduceLROnPlateau reacts to validation loss. With no validation
    set, the requested epoch count is used (subject to the optional time guard).
    """
    cfg = copy.deepcopy(cfg)
    validate_config(cfg)
    if not train_items:
        raise ValueError("Training set is empty")
    seed_everything(seed)
    rng = random.Random(seed)
    model = Tagger(cfg, len(vocab.word2id), len(vocab.char2id), len(vocab.tags)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5
    )
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    no_improvement = 0
    start_time = time.time()
    requested_epochs = int(cfg["epochs"])
    completed_epochs = 0
    for epoch in range(1, requested_epochs + 1):
        if max_seconds is not None and completed_epochs >= 2 and time.time() - start_time >= max_seconds:
            log(f"  time guard reached after {completed_epochs} epochs ({time.time() - start_time:.1f}s)")
            break
        epoch_start = time.time()
        train_loss = _train_batches(model, optimizer, train_items, vocab, cfg, device, rng)
        completed_epochs += 1
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": time.time() - epoch_start,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if val_items is not None:
            val = evaluate_model(model, val_items, vocab, cfg, device)
            record.update({
                "val_loss": val["token_loss"],
                "val_accuracy": val["accuracy"],
                "val_surface_oov_accuracy": val["surface_oov_accuracy"],
                "val_lookup_oov_accuracy": val["lookup_oov_accuracy"],
            })
            # Prefer the most accurate checkpoint; break ties with lower validation loss.
            improved = val["accuracy"] > best_accuracy + float(cfg["min_delta"])
            tied_better_loss = abs(val["accuracy"] - best_accuracy) <= float(cfg["min_delta"]) and val["token_loss"] < best_loss
            if improved or tied_better_loss:
                best_accuracy = float(val["accuracy"])
                best_loss = float(val["token_loss"])
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                no_improvement = 0
            else:
                no_improvement += 1
            scheduler.step(float(val["token_loss"]))
            record["lr"] = float(optimizer.param_groups[0]["lr"])
            log(
                f"  ep {epoch}/{requested_epochs} loss={train_loss:.4f} "
                f"val={val['accuracy'] * 100:.3f}% "
                f"OOV={val['surface_oov_accuracy'] * 100:.2f}% "
                f"lr={record['lr']:.2g} ({record['seconds']:.1f}s)"
            )
            history.append(record)
            if no_improvement >= int(cfg["patience"]):
                log(f"  early stopping: no validation improvement for {no_improvement} epochs")
                break
        else:
            record["val_loss"] = float("nan")
            record["val_accuracy"] = float("nan")
            log(f"  ep {epoch}/{requested_epochs} loss={train_loss:.4f} ({record['seconds']:.1f}s)")
            history.append(record)
        # Avoid starting a new epoch if the next one is unlikely to fit in the time budget.
        if max_seconds is not None and completed_epochs >= 2 and epoch < requested_epochs:
            elapsed = time.time() - start_time
            avg_epoch = elapsed / completed_epochs
            if elapsed + avg_epoch * 1.05 > max_seconds:
                log(f"  time guard predicts budget exhaustion; stopping at {completed_epochs} epochs")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    duration = time.time() - start_time
    info = {
        "seconds": duration,
        "epochs_run": completed_epochs,
        "best_epoch": best_epoch if val_items is not None else completed_epochs,
        "best_val_accuracy": best_accuracy if val_items is not None else None,
        "best_val_loss": best_loss if val_items is not None else None,
        "params": n_params(model),
        "history": history,
        "cfg": cfg,
    }
    return model, info


def predict_tags(
    model: Tagger,
    items: Sequence[dict[str, Any]],
    vocab: Vocab,
    cfg: dict[str, Any],
    device: torch.device,
    batch_size: int = 256,
) -> list[list[int]]:
    if not items:
        return []
    model.eval()
    out: list[list[int] | None] = [None] * len(items)
    order = sorted(range(len(items)), key=lambda index: len(items[index]["w"]))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = [items[index] for index in indices]
            word_ids, shape_ids, unique_chars, unique_ids, lengths, _ = collate(batch, vocab, cfg, device)
            logits = model(word_ids, shape_ids, unique_chars, unique_ids, lengths)
            decoded = model.decode(logits, lengths)
            for index, predicted in zip(indices, decoded):
                out[index] = predicted
    return [predicted if predicted is not None else [] for predicted in out]


def make_tagged_lines(
    sentences: Sequence[Sequence[str]], predictions: Sequence[Sequence[int]], tags: Sequence[str]
) -> list[str]:
    if len(sentences) != len(predictions):
        raise ValueError(f"Sentence/prediction count mismatch: {len(sentences)} vs {len(predictions)}")
    lines: list[str] = []
    for words, tag_ids in zip(sentences, predictions):
        if len(words) != len(tag_ids):
            raise ValueError(f"Token/tag count mismatch: {len(words)} words vs {len(tag_ids)} predictions")
        lines.append(" ".join(f"{word}/{tags[int(tag_id)]}" for word, tag_id in zip(words, tag_ids)))
    return lines


def load_checkpoint(path: str, device: torch.device) -> tuple[Tagger, Vocab, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    vocab = Vocab.from_state(checkpoint["vocab"])
    model = Tagger(cfg, len(vocab.word2id), len(vocab.char2id), len(vocab.tags)).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, vocab, cfg, checkpoint.get("metadata", {})


def save_checkpoint(
    path: str,
    model: Tagger,
    vocab: Vocab,
    cfg: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "cfg": copy.deepcopy(cfg),
            "vocab": vocab.state(),
            "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": metadata or {},
        },
        path,
    )
