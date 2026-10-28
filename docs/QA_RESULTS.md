# Software QA record

This record documents software tests, not model-quality results. Tiny-subset accuracies are not meaningful and must not be cited in the assignment report.

## Checks performed in the packaging environment

- **Python syntax:** all project scripts and tests passed `py_compile` after final edits.
- **Data contract:** passed on the provided full data files: 39,832 tagged training sentences / 950,028 tokens; 1,993 dev sentences / 47,633 tokens; 45 training tags; `dev.sents` and `dev.tags` match sentence-by-sentence and token-by-token.
- **Registered-variant stress test:** all 23 variants completed forward/backward training, validation, checkpoint save/load, and prediction on a 40-sentence slice of the actual `train.tags` file, on CPU. This is a software smoke test only.
- **Experiment CLI smoke:** `run_all.py --profile full --epochs 1 --max-sentences 40 --smoke-only` executed all 23 variants and generated plots plus Markdown/LaTeX report drafts without training on synthetic text.
- **Full-run code-path integration:** used a small subset of actual `train.tags` lines to exercise variant selection, the deployment refit, `run_tagger.py`, dev evaluation, diagnostics, figures, and report output. The resulting accuracy was deliberately not treated as a benchmark.
- **Assignment-style entry-point smoke:** trained a checkpoint on 60 real training sentences, ran `run_tagger.py` on `sample_test.sents`, and evaluated its output against the supplied `sample_test.out`. Output tokenization aligned; the low score is expected for a tiny, one-epoch-scale test and is not model performance.

## What was not performed here

- No full 23-variant training sweep on all 950,028 training tokens was run in this packaging environment.
- No run on the user's RTX 4060 or Google Colab was possible from this workspace.
- No claim is made that the peer report's 97.72% development accuracy is reproduced.
- The generated ACL LaTeX was structurally checked (document/table begin/end balance) but not compiled to PDF, because the official ACL style files are not installed here. Compile in the official ACL Overleaf template and visually inspect page count, figures, and tables.
