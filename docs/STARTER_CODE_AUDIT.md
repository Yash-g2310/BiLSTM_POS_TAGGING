# Starter-code and peer-report audit

This audit distinguishes what was present in the supplied starter code from what the peer report claims. It is not a claim that the peer report's results are wrong; they are simply not reproducible from the starter defaults without further configuration and a full run.

## What the supplied starter already implemented

The supplied `tagger_lib.py` implements a trainable lexical embedding, a character encoder (CNN or character LSTM), an optional 8-class shape embedding, a stacked BiLSTM, and a token-level output layer. No pretrained word vectors are loaded. Its default token input is 50 word + 40 character + 10 shape = 100 dimensions, which satisfies the PA2 input-dimension constraint. Its default recurrent hidden width is 256 per direction, with 2 LSTM layers. It has word dropout, dropout, gradient clipping, and early-stopping support in the training helper.

The starter `experiments.py` contains useful initial variant families for feature ablations, representation dimensions, model size, regularization, and optimization. However, it trains each variant against the full training set and scores it on `dev.tags`. Using the official development labels to choose variants can overfit the development set. The results are appended to a CSV without a shared internal validation protocol or robust resume/configuration tracking.

The starter `build_tagger.py` holds out a set of training sentences, but builds its vocabulary from the full input corpus before that holdout is excluded. That makes rare lexical items from the held-out portion visible to vocabulary construction. The current project builds the vocabulary on the optimization partition for model selection and keeps `dev.tags` outside hyperparameter selection.

## Why the reported 97.72% needs to be verified

The attached peer report describes an 80-D lexical + 20-D character representation, 16-D character embeddings, CNN widths 3 and 5, 192 recurrent units per direction, batch size 128, learning rate 0.003, dropout 0.35, and 16 epochs. The supplied code's default is materially different: 50-D word + 40-D character + 10-D shape, 25-D character embeddings, a 256-unit BiLSTM, batch size 64, learning rate 0.002, different dropout settings, and a single width-3 CNN. The starter's 8-way shape embedding and generic unknown-word handling also do not reproduce the report's described 81 lexical-signature scheme.

The new runner therefore includes a report-like 80/20/192 configuration for controlled comparison, but does not label it an exact recreation of the report's full preprocessing. The 97.72% score in the peer report is recorded as a comparison point only until a run on the supplied `train.tags` / `dev.sents` / `dev.tags` files independently reproduces it.

## What the revised project adds

- Validates sentence/token alignment and the 45-tag inventory before spending training time.
- Uses a fixed sentence-level validation split from the training corpus to compare variants; the released development set is evaluated after selection rather than used for tuning.
- Registers 23 experiments covering core feature ablations, CNN-vs-character-LSTM, uni-vs-bidirectional recurrence, CRF-vs-softmax decoding, case sensitivity, input-dimension allocation, recurrent size/depth, dropout, word dropout, learning rate, batch size, and label smoothing.
- Records per-run configs, random seed, data hashes, parameter counts, actual epoch counts, validation metrics, and runtime.
- Saves a full-training-data refit of the selected model, creates the required `dev.out`, verifies alignment, and generates token-level diagnostics, plots, and editable Markdown/ACL-LaTeX report drafts.
- Includes CPU smoke/stress tests using real subsets of the supplied corpus rather than generated sentences.

## Remaining manual items

1. Run the full suite on the target hardware; smoke results are software checks only.
2. Compare against your group's PA1 HMM output on exactly the same dev files. The HMM scores quoted in the peer report have not been independently verified by this project.
3. Enter the actual group member names/roll numbers and record the hardware used for the full run.
4. Inspect the generated ACL PDF on Overleaf and enforce the assignment's 4–6 page limit before printing.
5. Confirm course policy before publishing the assignment data or handout publicly on GitHub.
