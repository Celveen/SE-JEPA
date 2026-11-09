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

- `sec/`: abstract, main text, AI-use statement, and appendices. Section 3.1 motivates cross-episode replay, with the paired weight-inheritance diagnostic in main Table 2. Section 4.1 describes the evaluation setup; Section 4.5 compares the read and write protections and tests how to carry experience between episodes. The main text includes the same-checkpoint component comparison. Main Table 3 reports the same-checkpoint components and fixed slow learning rates. Appendix D gives the familiarity ablation and protocol details.
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
python tools/build_condition_gains.py
python tools/build_deform_appendix.py
python tools/build_reference_rollouts.py
python tools/build_mixed14_assets.py
python tools/audit_pusht.py
python tools/package_overleaf.py
```

These optional scripts use NumPy, Matplotlib, Pillow, and PyMuPDF. The first script
otherwise restores older panels; the later scripts apply the current style
and original simulator frames. Figure 1 combines the recurring-shift schematic and
the held-out GD results in one column. SE-JEPA uses the three-chain GD
mean. DINO-WM and LeWM use the seven-shape PushObj results for seed 2000.
The PushT panel uses the paired visual evaluation of Frozen, AdaJEPA, and
SE-JEPA, plus the November 8 GD baseline archive: DINO-WM 114/300 (38.00%)
and LeWM 118/300 (39.33%) across six shifts. All 14 baseline NPZ files were
checked against the archived hashes and counts. Methods are ordered by increasing PushObj success. The episode-local
baseline is labeled adaJEPA in the lower chart and One-round adaptive JEPA
in the upper schematic of Figure 1.
The complete editable SVG is
`figures/source/teaser_combined.svg`. The author's supplied Illustrator SVG is
`figures/source/teaser_combined_original.svg`. `main.tex` includes the ready-to-use
`figures/teaser_combined.pdf`.

To edit the bar data and export the full figure, update `data/teaser_bars.csv` and run:

```sh
python tools/plot_teaser.py --data data/teaser_bars.csv --output-dir figures
python tools/combine_teaser.py --template figures/source/teaser_combined.svg --bars-dir figures --output-dir figures
cp figures/teaser_combined.svg figures/source/teaser_combined.svg
```

The combined exporter uses CairoSVG and PyMuPDF. `tools/build_condition_gains.py`
regenerates the main per-condition comparison from the unrounded held-out rates and
checks each value against the saved success and episode counts. Figure 7 uses a
single-column layout with PushObj above PushT.
Figures use bold
method labels for SE-JEPA and regular labels for the baselines. When comparing
methods, line plots also give SE-JEPA a heavier stroke. GD and CEM share the
same weight in plots where both planners use SE-JEPA.

DINO-WM, DreamerV3 and IRIS use the episode-verified VIS50 results from the
`baselines` branch (original paths, immutable commit and checksums in
`data/vis50/vis50_sources.json`). The latter two methods use offline predictors
for CEM. The SE-JEPA PushObj cells in the main table use fifth-round online CEM
performance across S7t3, S7t3b, and S7t3s700 (80.4% All, 73.1% Unseen,
12.99 replans). The DreamerV3 and IRIS PushObj Unseen scores use the published
two-shape protocol of Zhou et al. Their paper also supplies the rope and
granular CDs for DreamerV3, IRIS, R3M, and DINO CLS. The two local frozen
deformable entries refer to the same DINO-WM model. Section 4.1 and
Appendix C describe these protocols. `tools/audit_pusht.py` checks the
corresponding main-table cells against the visual evaluations.

The joint familiarity ablation gives 836/1,050 successes with CEM and
873/1,050 with GD. [Ablation archive](https://github.com/Celveen/SE-JEPA/tree/3faad041417c6f2a22df2ad62822c34fd075c4bb/experiments/ablation_archive_20261106)

Figure 1 source counts are recorded in `data/reporting_overrides.json`.
[GD baseline archive](https://github.com/Celveen/SE-JEPA/tree/eec0ba00fbb945c4dc70bd4804035f845b67a463/research_snapshots/2026-11-07/baseline_metadata)

The main qualitative comparison selects real examples where Frozen and
AdaJEPA fail and SE-JEPA succeeds. Selection and source hashes are recorded
with the assets. Rope/Granular examples are from the third pass, while their
main-table aggregate results are from the fourth pass; the caption distinguishes
these. The appendix retains the four supplied reference panels: square, red-agent,
rope example 1, and granular example 2. It also keeps rope example 2 and granular
example 1. All six panels come from native Drive PDFs, preserving the original
frames, labels, loss curves and explanatory notes. The deformable cases are
cropped at their original panel boundaries; the rigid-object panels retain their
complete frame strips and vector loss curves. `tools/build_reference_rollouts.py`
reproduces them, with source hashes, image checks and crop bounds in
`data/revision_assets/reference_rollout_sources.json`. The older reformatted
views remain in the research archive but are excluded from the clean upload.

Before camera-ready release, supply author details, change the review option
in `main.tex`, and reconcile the AI-use statement with all authors' tool use.

## Current layout and bibliography

The main comparison table is on page 5, the main result figures on pages 5–8,
and Conclusion with Future work closes the eight-page main text on page 8. Result floats
are declared early in `sec/5_result_floats.tex`; a float barrier before Conclusion
keeps them ahead of the closing section. The bibliography has 42 cited entries, including the original ResNet paper,
with primary-source checks recorded in `data/reference_verification.json`.
Appendix A reports training and evaluation on systems with 8 NVIDIA GeForce
RTX 5090 GPUs and 8 NVIDIA A100 GPUs, as supplied by the authors.

## Backbone and learning-progress revision

PushObj and PushT use task-trained ResNet-based world models; rope and granular
use the DINO-WM backbone. Figure 6 in the main text combines five-round unseen
shape and color-shift success with four-pass rope and granular CD. The lines show means. The numerical archive also contains sample SD
across the three learning chains and SEM over 50 fixed deformable instances. The color plot aggregates only red agent, red object,
and red goal, using archived successes and counts rather than seven-condition
averages. `tools/build_learning_progress.py` writes its numerical audit to
`data/learning_progress.json`. The fourth-pass deformable values come from the
existing manuscript snapshot; the archived Drive qualitative package remains
at the third pass. The Future work observation about later prediction/behavior
saturation was provided by the authors and is not presented as a new quantified
experiment.

## November 8 checkpoint and GD revision

PushObj and PushT AdaJEPA benchmarks use their official released checkpoints;
`data/checkpoint_audit.json` records the provenance and routed result sources.
The same-checkpoint ablation is labeled Episodic reset. Rope/granular values
remain unchanged at the author's request. Figure 5 now uses GD curves and
Figure 6 marks Success (%) up and Chamfer distance down. The main method has
five numbered equations, including familiarity and fingerprint similarity.


## Mixed 14-domain stream and original reference panels

Main Section 4.5 (page 7) and Figure 8 (page 8) present continual improvement
with one memory spanning seven shape domains and seven visual conditions.
The three-panel figure shows five-round overall, shape and visual success,
averaging GD and CEM equally. Frozen and AdaJEPA are horizontal references
from their archived evaluations. Shading gives sample SD across six actual
planner-by-chain or planner-by-seed outcomes; the baselines were not run for
five continual rounds. `data/mixed14_planner_average.json` records these
aggregates and the original audit checksum. Appendix D retains Tables 8–9
(overall and per-domain outcomes) and the detailed experimental protocol. This is online adaptation on repeated target pairs, separate from the
held-out benchmarks. The mixed-stream Frozen/AdaJEPA baselines use the official
shape-shift checkpoint, while SE-JEPA uses the TLZ+ epoch-3 checkpoint.

`data/mixed14_audit.json` records an independent recomputation of all 588 result
files (29,400 episodes), 420 state audits and 414 consecutive state links. It
contains all plot/table values and source checksums. Reproduce it using
`python tools/verify_mixed14.py /path/to/the/supplied/mixed14/archive`, then run
`tools/build_mixed14_assets.py`. The supplied eight-page report is preserved as
`figures/source/mixed14_author_report.pdf`; raw experiment archives and model
weights are not added to the Overleaf bundle.

The combined paper has 20 pages, including eight pages of main text. Appendix E
restores the reference panels' full narrative, explaining successful replans,
final CD, action-replay provenance and the interpretation of prediction losses.


## PushObj CEM main-table reporting

The main table reports the selected fifth-round online learning results:
288/350, 280/350, and 276/350 successes, pooling to 844/1,050 (80.38%).
Unseen shapes contribute 329/450 (73.11%); all-episode mean replans are 12.99,
counting failures as 30. `data/pushobj_cem_online/audit.json` records the 21
source NPZ files and immutable Git hashes. `tools/pushobj_online.py` rechecks
these bundled files whenever `tools/build_paper_assets.py` regenerates the table.

The paired post-learning tables and Figure 7 retain their separate evaluation
results (CEM All 79.08%). The main caption and Appendix B identify the different
online and baseline evaluation target sets. The approximately 15-point
held-out unseen-shape comparison in the main text refers to the paired appendix
table, not the online row. The exact-state overlap audit is archived alongside
the online results and noted in Appendix B.

## Matched DINO-WM examples in Figure 3

Figure 3 uses 90% of the text width and no longer includes the in-panel Goal
explanation. The I-shape row adds DINO-WM GD on the same initial and goal states;
its failure at 30 replans agrees with the archived result. The source actions
were replayed in the original simulator. Both endpoint reference images match
the existing case, and the same procedure reproduces all 31 Frozen frames and
states exactly. Original results, simulator sources, frames and a portable
`replay.py` are retained in `data/revision_assets/dinowm_shape_I/`.

The red-object row now uses an exactly matched target shared by DINO-WM GD
(sample 31) and the mixed 14-domain experiment (sample 30). Frozen and AdaJEPA
use seed 100 and the official shape-shift checkpoint; SE-JEPA uses round five
of chain two. This is a selected mixed-stream illustration, not a replacement
for the standard visual benchmark. All three baselines fail within 150 executed
environment steps, while SE-JEPA succeeds after 30. The visual DINO-WM run
executes 25 steps per replan and the others five. Both figure rows label actual
MPC replans: visual DINO-WM fails at six, other baselines fail at 30, and
SE-JEPA succeeds at six in both examples. The caption states the frequency
difference. The CEM main-table step units are unchanged.

`data/revision_assets/dinowm_visual_mixed/` stores the visual replay and matching
audit. Provenance files record source hashes and validation; the earlier visual
case remains in `selected_cases/visual_GD_red_object/`. The clean upload contains
only the updated figure PDF; no simulator dependencies are needed for Overleaf.
