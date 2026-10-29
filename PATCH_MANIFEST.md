# Update-only patch manifest

The ZIP is **flat-root** and should be extracted at the root of `POS_Tagger_PA2_Complete`. It contains only these classes of updates:

- `.gitignore` — leaves measured `results/` trackable but ignores `.venv`, model checkpoints, course data, and temporary outputs.
- `README.md` — concise project overview, architecture image, actual results, HMM aggregate baseline, and report-link placeholder.
- `docs/figures/` — architecture and evaluation-protocol diagrams in PNG/SVG.
- `report/` — editable ACL LaTeX, bibliography, four main-paper figures, Draw.io source diagrams, and a 5-page local proof PDF.
- `results/` — the actual 23-variant results, metric JSON/CSVs, predictions, history, logs, and run figures; no model checkpoint or course dataset.

No `tagger_lib.py`, trainer, inference implementation, `.venv`, `data/*.tags`, `data/*.sents`, or `.pt` model is included in this patch.
