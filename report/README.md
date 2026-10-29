# PA2 Report Files

- `main.tex`: editable official ACL-style source.
- `references.bib`: bibliography.
- `figures/`: images referenced by the source.
- `drawio/`: editable diagrams.net files.
- `FINAL_REPORT_LOCAL_PROOF.pdf`: five-page local proof; compile on the official ACL Overleaf template before printing.

Upload `main.tex`, `references.bib`, and files from `figures/` to the official ACL Overleaf template, preserve the `figures/...` paths, and keep its `acl.sty`/`acl_natbib.bst`. Check that the compiled report remains 4–6 pages.

Authors: Yash Gupta (23118081), Adarsh Maurya (23125004). Yash's PA1 HMM: 96.4122% accuracy; Adarsh's separate HMM is intentionally pending. No aligned PA1 predictions were supplied, so only aggregate comparison is included.

## Applying this update to the existing repository

The update patch includes the new README, `.gitignore`, report assets, architecture images, and measured `results/` outputs. It intentionally excludes model checkpoints and course data. Overlay it on your project root. The `.gitignore` will not remove previously tracked data or checkpoint files from Git; check `git ls-files` before committing. If prohibited data or model files are already tracked, remove those specific paths from the index with `git rm --cached <path>` (without deleting your local working copy), then commit. Keep all measured `results/` outputs tracked.
