#!/usr/bin/env python3
"""Validate the provided PA2 data and assignment IO contract."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis_tools import validate_parallel_data
from tagger_lib import DEFAULT_CFG, Vocab, norm, read_sents, read_tagged, validate_config


def main() -> None:
    data = ROOT / "data"
    train = read_tagged(str(data / "train.tags"))
    dev_sents = read_sents(str(data / "dev.sents"))
    dev_tags = read_tagged(str(data / "dev.tags"))
    sample_sents = read_sents(str(data / "sample_test.sents"))
    sample_out = read_tagged(str(data / "sample_test.out"))
    validate_parallel_data(dev_sents, dev_tags)
    validate_parallel_data(sample_sents, sample_out)
    train_tags = {tag for _, tags in train for tag in tags}
    dev_tags_set = {tag for _, tags in dev_tags for tag in tags}
    assert dev_tags_set <= train_tags
    assert len(train) == 39832, f"Unexpected train sentence count: {len(train)}"
    assert len(dev_sents) == 1993 and len(dev_tags) == 1993
    assert len(train_tags) == 45
    cfg = dict(DEFAULT_CFG)
    validate_config(cfg)
    dim = cfg["word_dim"] + cfg["shape_dim"] + (cfg["char_out"] if cfg["char_type"] != "none" else 0)
    assert 0 < dim <= 100
    vocab = Vocab(train[:20], tag_inventory=train_tags)
    assert vocab.encode(["UnseenWord_123", "-LRB-", "can't"])['wid'].shape == (3,)
    assert norm("AbC123") == "abc000"
    # Both forms occur in the real training corpus; test the lookup mode on observed keys.
    case_insensitive = Vocab(train, tag_inventory=train_tags)
    case_sensitive = Vocab(train, tag_inventory=train_tags, case_sensitive_lookup=True)
    assert case_insensitive.encode(["IN"])["wid"][0] == case_insensitive.encode(["in"])["wid"][0]
    assert case_sensitive.encode(["IN"])["wid"][0] != case_sensitive.encode(["in"])["wid"][0]
    assert Vocab.from_state(case_sensitive.state()).case_sensitive_lookup is True
    print("PASS: file parsing/alignment, 45-tag inventory, normalized and case-sensitive lookup, checkpointable vocabulary, no pretrained vectors, and <=100-D config check.")
    print(f"train={len(train)} sentences / {sum(len(w) for w, _ in train)} tokens; dev={len(dev_sents)} sentences / {sum(map(len,dev_sents))} tokens")


if __name__ == "__main__":
    main()
