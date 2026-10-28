# Final 4–6 page ACL report checklist

Run `python run_all.py --profile full --device auto` and inspect `results/report_draft.md`, `results/report_draft.tex`, and `results/figures/`. The generated LaTeX is an editable starting point; it expects the official ACL style files from the ACL Overleaf template.

Before printing, verify all of the following:

1. Replace both group-member placeholders with the correct full names and roll numbers.
2. Report only results present in the latest `results/` folder and keep the config in the paper aligned with the model checkpoint metadata.
3. Add the verified PA1 HMM accuracy from the group's PA1 model, measured on exactly `dev.sents` / `dev.tags`. The peer report's HMM figures are not independently verified here.
4. Include the architecture figure, ablation comparison, learning curves, and at least one error-analysis figure/table; use the generated CSVs to provide concrete examples.
5. Explain both OOV definitions and how unknown tokens retain character/shape input.
6. Include parameter count, actual epoch count, `build_tagger.py` training runtime, `run_tagger.py` inference runtime, PyTorch/NumPy version, and hardware from the final run.
7. State that the full ablation suite uses one seed unless you also run seed replications. Do not describe dev-set accuracy as blind-test accuracy.
8. Keep the final report within the assignment's 4–6 page requirement, compile with the official ACL style, render the PDF, and visually inspect every page for table/figure clipping.
