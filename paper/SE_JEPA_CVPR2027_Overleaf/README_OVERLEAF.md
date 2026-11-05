# SE-JEPA — CVPR 2027 manuscript

Upload `SE_JEPA_CVPR2027_Overleaf.zip` with **New Project → Upload Project**.
Select `main.tex` and the **pdfLaTeX** compiler. The project uses BibTeX and the
unchanged official CVPR 2027 style in anonymous review mode.

`main.tex` contains the paper, AI-use statement, references, and appendices A–E.
The appendix starts on a new page for reviewing the complete manuscript.
`supplement.tex` also builds the appendix separately; compile `main.tex` first
so its cross-reference file is available. No shell escape, Python, external
image links, SVG conversion, or custom fonts are required by Overleaf.

## Source organization

- `sec/`: abstract, main text, AI-use statement, and appendices. The deployment
  setting and inheritance diagnosis in `sec/3_problem.tex` are included within
  the SE-JEPA method section rather than as a separate section.
- `figures/`: ready-to-compile figures; author-supplied vector originals are
  preserved in `figures/source/`.
- `tables/`: generated numerical rows.
- `references.bib`: bibliography including the compared methods and classic
  world-model, JEPA, adaptation, and continual-learning work.
- `data/`: reporting snapshots, archived episode results, selected simulator
  frames, provenance manifests, and audit results.
- `tools/`: optional regeneration and packaging scripts.
- `AUTHOR_NOTES.md`: author decisions and pending data confirmation; not part
  of the compiled paper.

## Build and check

```sh
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

The latest local checks are recorded in `compile_checks.json`. Recheck the
main-text page limit and all figure placements after edits. The combined PDF
is for whole-paper review; prepare submission files according to venue rules.

## Figure and data regeneration

Run these scripts in order when regenerating all figures:

```sh
python tools/build_paper_assets.py
python tools/restyle_paper_figures.py
python tools/build_revision_assets.py
python tools/audit_pusht.py
python tools/package_overleaf.py
```

These optional scripts use NumPy, Matplotlib, and Pillow. The first script
otherwise restores older panels; the later scripts apply the current style
and original simulator frames. The first-page chart uses blue-gray shades,
with SE-JEPA the most saturated. Other statistical plots follow the supplied
Drive palette. Simulator pixels and the author's architecture diagram retain
their original colors.

DINO-WM's three PushT main-table cells remain blank pending author confirmation.
At the author's request its existing first-page bar is displayed normally at
57.0; this does not constitute verification. The exception is recorded in
`data/reporting_overrides.json`. Core paired PushT results are checked against
archived episode data by `tools/audit_pusht.py`.

The main qualitative comparison selects real examples where Frozen and
AdaJEPA fail and SE-JEPA succeeds. Selection and source hashes are recorded
with the assets. Rope/Granular examples are from the third pass, while their
main-table aggregate results are from the fourth pass; the caption distinguishes
these. The appendix retains additional complete trajectories.

Before camera-ready release, supply author details, change the review option
in `main.tex`, and reconcile the AI-use statement with all authors' tool use.

## Current layout and bibliography

The main comparison table is on page 5, the main result figures on pages 5–6,
and Conclusion with Future work closes the main text on page 7. Result floats
are declared early in `sec/5_result_floats.tex`; a float barrier before Conclusion
keeps them ahead of the closing section. The bibliography has 40 cited entries,
with primary-source checks recorded in `data/reference_verification.json`.
Appendix A reports training and evaluation on systems with 8 NVIDIA GeForce
RTX 5090 GPUs and 8 NVIDIA A100 GPUs, as supplied by the authors.
