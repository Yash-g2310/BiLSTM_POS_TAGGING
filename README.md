# DAL-585 PA2 — Neural Part-of-Speech Tagger

A reproducible PyTorch POS-tagger project for the DAL-585 Programming Assignment 2. It includes the required assignment entry points, an ablation runner, a CPU-compatible smoke/stress suite, automatic development-set error analysis, plots, and editable report drafts.

## What the implementation does

- Learns all embeddings from `train.tags`; **no pretrained word vectors are used**.
- Uses a trainable lexical embedding, a character encoder (`cnn`, `lstm`, or disabled), and an optional compact token-shape embedding.
- Uses a BiLSTM encoder by default, with either a token-softmax or optional linear-chain CRF decoder.
- Keeps the concatenated per-token input representation at **100 dimensions or fewer**. The default is `50 word + 40 character + 10 shape = 100`.
- Normalizes lexical lookup by lowercasing and mapping digits to `0`, but character features use the original spelling. A word whose normalized form is unseen maps to the unknown ID; the character and shape paths still receive the token.
- Uses a deterministic sentence-level holdout from `train.tags` for model selection, dropout, frequency-aware word dropout, gradient clipping, a learning-rate scheduler, and early stopping.
- Keeps `dev.tags` out of hyperparameter selection. The provided development labels are used only for final evaluation and error analysis in the full pipeline.

The supplied peer report describes an 80-D word + 20-D character configuration with 192 hidden units and 16-D character embeddings. The starter code defaults to a different 50/40/10 configuration. This project tests both the general default and a peer-report-like `paper_spec_80_20_192` variant, uses multi-width character convolutions (3 and 5), and records the **actual configuration used by each run** to avoid a mismatch between implementation and report.

## Data files

The `data/` directory contains the files supplied for PA2:

- `train.tags` — 39,832 tagged sentences, 950,028 tokens.
- `dev.sents` — 1,993 tokenized sentences.
- `dev.tags` — gold labels aligned token-by-token with `dev.sents`.
- `sample_test.sents` and `sample_test.out` — the assignment's sample input and example output.

Dataset and assignment handout redistribution may be restricted. The complete runnable ZIP includes these files for convenience, but `.gitignore` excludes the data by default. Confirm your course rules and obtain permission before pushing course files or dataset contents to a public GitHub repository.

## Recommended environment

**Use your RTX 4060 for the full ablation suite.** It is much more practical than CPU-only training. Google Colab GPU is the fallback and matches the assignment's stated test environment. CPU mode is supported for smoke tests, smaller experiments, and inference, but the complete 23-variant suite can take a long time on CPU.

The assignment specifies Python 3.10, PyTorch 2.11+cu128, and NumPy 2.1.3. PyTorch's official 2.11 installation supports a CUDA 12.8 wheel; the current PyTorch installation guide may offer newer CUDA wheels too. Use a compatible wheel for the installed NVIDIA driver (see the links below).

### Windows + RTX 4060 (PowerShell)

From the unzipped project directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==2.1.3 matplotlib pandas
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

If PowerShell blocks activation, see the standard Python `venv` documentation or run the commands from Command Prompt with `.venv\Scripts\activate.bat`. Do not install a CUDA toolkit separately just to use the PyTorch wheel; the NVIDIA driver still needs to be installed and current.

### Google Colab

Select a GPU runtime, upload/unzip the project, and run from the project root. If Colab's preinstalled versions do not match PA2's requested environment, install compatible versions first:

```bash
python -m pip install numpy==2.1.3 matplotlib pandas
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Some Colab runtimes may ship with another PyTorch version. Check the printed version rather than assuming a GPU is active.

### CPU-only

```bash
python -m pip install numpy==2.1.3 matplotlib pandas
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The implementation automatically picks CPU if CUDA is unavailable. If the full experiment suite is too slow, run `--profile quick` or `--profile standard` rather than pretending a one-epoch smoke run is a meaningful accuracy evaluation.

## Required assignment commands

Run these from the project root, where `data/` is present. The commands work with the included relative data paths, and you can also pass your own paths.

```bash
python build_tagger.py data/train.tags
python run_tagger.py data/dev.sents dev.out
python evaluate.py dev.out data/dev.tags
```

The instructor's usage forms work too:

```bash
python build_tagger.py train.tags
python run_tagger.py dev.sents dev.out
python MeasurePOSTaggerAccuracy.py dev.out dev.tags
```

For the second form, copy `data/train.tags`, `data/dev.sents`, `data/dev.tags`, and `MeasurePOSTaggerAccuracy.py` into the current working directory first. `MeasurePOSTaggerAccuracy.py` checks alignment and prints the assignment-style `Accuracy= ...` line; `evaluate.py` reports both token accuracy and exact-sentence accuracy.

`build_tagger.py` writes `tagger_model.pt` to the current working directory. The `run_tagger.py` inference command loads that checkpoint and preserves token order and blank lines.

## Full experiment pipeline — one command

This is the main command to generate experiment results, full-data model, development predictions, charts, error analysis, and report drafts:

```bash
python run_all.py --profile full --device auto
```

`--profile full` runs 23 registered variants: feature ablations, character encoder comparison, forward-only vs bidirectional recurrence, softmax vs CRF, dimension allocations, model size/depth, dropout, case-sensitive lexical lookup, a peer-report-like 80/20/192 configuration, word dropout, learning rate, batch size, and label smoothing. Experiments share the same training-derived validation split and seed. After selection, the chosen configuration is trained on all of `train.tags` for the selected epoch count, `tagger_model.pt` is written in the working directory, and `dev.out` is generated/evaluated.

Useful alternatives:

```bash
# A smaller but useful first experiment pass (12 variants)
python run_all.py --profile standard --device auto

# Quick architecture sanity run (4 variants, 4 epochs by default)
python run_all.py --profile quick --device auto

# Optional time guard for each individual variant
python run_all.py --profile full --device auto --max-seconds-per-model 600

# Smoke the entire experiment pipeline on 40 actual training sentences, one epoch.
# This writes only to results_smoke and does NOT fit the final model or write dev.out.
python run_all.py --profile full --epochs 1 --max-sentences 40 --smoke-only --output-dir results_smoke --device cpu
```

`run_all.py` writes `tagger_model.pt` in the current working directory after it selects a model on validation and refits it on all of `train.tags`. It also writes `dev.out` and checks it against `dev.tags`. **That is the recommended all-in-one submission run.** Running `build_tagger.py` afterward intentionally overwrites `tagger_model.pt` with the standalone validation-selected checkpoint, so do not run it after the full pipeline unless that is desired.

The final full-data refit has a soft 14-minute budget by default (`--final-time-budget 840`), to stay aligned with the assignment's 15-minute training target. If the environment is unusually slow, the time guard stops between epochs and records the actual epoch count.

## Smoke, stress, and data checks

These checks use subsets of the *provided* `train.tags`, not synthetic fake sentences:

```bash
python tests/data_contract_test.py
python tests/stress_test.py --data-dir data --max-sentences 60 --epochs 1 --device cpu
```

The stress suite instantiates every registered architecture and decoder, runs forward/backward training, evaluates its validation output, round-trips a checkpoint, and predicts tags. Its tiny-data accuracy numbers are **software smoke-test values only and must not be reported as model performance**.

For a fast end-to-end CLI test that writes to a separate folder:

```bash
python build_tagger.py data/train.tags --config tests/smoke_config.json --max-sentences 60 --time-budget 60 --model-out tmp_smoke/tagger_model.pt
python run_tagger.py data/sample_test.sents tmp_smoke/sample_test.out --model tmp_smoke/tagger_model.pt
python evaluate.py tmp_smoke/sample_test.out data/sample_test.out
```

This test checks I/O and checkpoint compatibility, not accuracy; the model is trained for one epoch on only 60 sentences.

## Important outputs

| Path | Contents |
|---|---|
| `tagger_model.pt` | Final deployment checkpoint created by `run_all.py` or `build_tagger.py` |
| `dev.out` | Tagged output in the required `word/TAG` format |
| `results/experiments.csv` | Ablation/hyperparameter results, one row per variant, including the model-specific lookup-OOV metric |
| `results/history.csv` | Training/validation curves by epoch and variant |
| `results/run_manifest.json` | Seed, versions, split settings, device, input-file hashes |
| `results/dev_metrics.json` | Dev token accuracy, exact-sentence accuracy, and OOV metrics |
| `results/dev_token_predictions.csv` | Gold/prediction for every token, with OOV indicators |
| `results/per_tag_metrics.csv` | Support, recall/accuracy, precision, F1, and errors by tag |
| `results/confusion_pairs.csv` | Directed gold-tag → predicted-tag confusion counts |
| `results/error_examples.csv` | Mis-tagged tokens with sentence context |
| `results/figures/` | Architecture diagram, learning curves, comparison plots, tag confusion and corpus figures |
| `results/report_draft.md` | Results-filled editable Markdown report draft |
| `results/report_draft.tex` | ACL-style editable LaTeX report source |
| `results/references.bib` | Bibliographic references for the generated LaTeX draft |
| `results/runtime_summary.json` | End-to-end and final-fit runtime metadata |

The LaTeX draft expects the official ACL style files (`acl.sty` and `acl_natbib.bst`). Import the official ACL Overleaf template, then replace its main `.tex` and bibliography with the generated draft and `references.bib`. Before submitting, add the correct group names/roll numbers and the verified PA1 HMM comparison. The included peer report mentions HMM figures of 96.34% and 96.32%, but those are not independently verified by this code.

## Avoiding overfit and honest reporting

- Select models on the fixed training-derived validation split, not `dev.tags`.
- Do not choose hyperparameters by repeatedly optimizing the released dev score.
- Report both token accuracy and exact-sentence accuracy, and separate known/OOV performance.
- Inspect per-tag and directed confusion tables instead of only reporting one headline score.
- The full suite currently uses one seed to control compute. For strong research claims, rerun the best 1–3 variants using multiple seeds and report mean and standard deviation.
- The neural dev score is not a blind-test score. The final grade is measured on a separate blind test set.

## References / research starting points

- dos Santos, C. N. and Zadrozny, B. (2014). [Learning Character-level Representations for Part-of-Speech Tagging](https://proceedings.mlr.press/v32/santos14.html). ICML/PMLR.
- Ma, X. and Hovy, E. (2016). [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://aclanthology.org/P16-1101/). ACL.
- Marcus, M. P., Santorini, B., and Marcinkiewicz, M. A. (1993). [Building a Large Annotated Corpus of English: The Penn Treebank](https://aclanthology.org/J93-2004/). *Computational Linguistics*.
- [PyTorch 2.11 release / installation notes](https://pytorch.org/blog/pytorch-2-11-release-blog/).

## GitHub checklist

A GitHub-ready `.gitignore` is included. To prepare a repository:

```bash
git init
git add README.md *.py tagger_lib.py experiment_runner.py analysis_tools.py requirements.txt .gitignore docs tests
# Do not force-add course dataset/handout/model artifacts until the course policy permits it.
git status
git commit -m "Add reproducible neural POS tagging experiments"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Create the empty GitHub repository first and replace the placeholder remote. Keep it private until both group members agree on attribution and course rules permit public sharing. The course submission ZIP should remain self-contained even if the GitHub repo excludes the dataset.
