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

- `sec/`: abstract, main text, AI-use statement, and appendices. Section 3.1 motivates cross-episode replay, with the paired weight-inheritance diagnostic in Appendix D. Section 4.1 describes the evaluation setup.
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
python tools/build_learning_progress.py
python tools/audit_pusht.py
python tools/package_overleaf.py
```

These optional scripts use NumPy, Matplotlib, Pillow, and PyMuPDF. The first script
otherwise restores older panels; the later scripts apply the current style
and original simulator frames. The first-page chart uses blue-gray shades,
with a bright-blue, dark-outlined and hatched SE-JEPA bar. Figures use bold
method labels for SE-JEPA and regular labels for the baselines. When comparing
methods, line plots also give SE-JEPA a heavier stroke. GD and CEM share the
same weight in plots where both planners use SE-JEPA.

DINO-WM, DreamerV3 and IRIS use the episode-verified VIS50 results from the
`baselines` branch (original paths, immutable commit and checksums in
`data/vis50/vis50_sources.json`). The latter two methods use offline predictors
for CEM. The main table reproduces the published two-shape PushObj success rates from Table 4 of Zhou et al. in a dedicated column. Results from our seven-shape protocol occupy separate columns. Italic rope/granular CD entries
are published measurements; the two local frozen deformable entries refer to
the same DINO-WM reference. `tools/audit_pusht.py` checks VIS50 and the separate
paired evaluation against the corresponding main-table cells.

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
keeps them ahead of the closing section. The bibliography has 42 cited entries, including the original ResNet paper,
with primary-source checks recorded in `data/reference_verification.json`.
Appendix A reports training and evaluation on systems with 8 NVIDIA GeForce
RTX 5090 GPUs and 8 NVIDIA A100 GPUs, as supplied by the authors.

## Backbone and learning-progress revision

PushObj and PushT use task-trained ResNet-based world models; rope and granular
use the DINO-WM backbone. Figure 7 in the main text combines five-round unseen
shape and color-shift success with four-pass rope and granular CD. The lines show means. The numerical archive also contains sample SD
across the three learning chains and SEM over 50 fixed deformable instances. The color plot aggregates only red agent, red object,
and red goal, using archived successes and counts rather than seven-condition
averages. `tools/build_learning_progress.py` writes its numerical audit to
`data/learning_progress.json`. The fourth-pass deformable values come from the
existing manuscript snapshot; the archived Drive qualitative package remains
at the third pass. The Future work observation about later prediction/behavior
saturation was provided by the authors and is not presented as a new quantified
experiment.
