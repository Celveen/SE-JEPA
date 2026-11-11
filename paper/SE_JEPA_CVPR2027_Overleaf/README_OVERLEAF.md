# SE-JEPA — CVPR 2027 manuscript

## Author-revised Figure 2 (2026-11-10)
Figure 2 now uses the newly supplied `main figue.svg`, archived verbatim as
`figures/source/architecture_author_20261110.svg`. The author's font colors,
typefaces, positions, and connections are retained. Only the Square/T test-time
images and their labels are replaced with the same verified Rope/Granular
simulator observations used previously. Both the displayed SVG and its embedded
draw.io document are updated. The PDF uses vector outlines from the embedded
fonts so no local font substitution is needed. The editable SVG retains text.
Rebuild with `python tools/update_overview_tasks.py` (CairoSVG/Cairo, fonttools,
Brotli, uharfbuzz, PyMuPDF). Provenance records the original images and SVG hashes.
Only page 3 changes in the rendered paper; all other pages remain identical.


## Overleaf sync and fixed-weight clarification (2026-11-10)
Downloaded the current Overleaf project and compared every manuscript file.
The only text change relative to the previous local release was in the abstract;
the teacher's source-code availability sentence is retained. The original
pretrained weights are now explicitly fixed, fast adaptation uses working copies,
and slow memory is learned in separate predictors. This small clarification is
consistent across the abstract, introduction, related work, method and appendices.
All numerical claims and figures are unchanged. The main text remains eight pages,
with 2.43 trailing lines. The original supplied abstract remains an archived source;
the current abstract supersedes the earlier verbatim-copy status.
The sync record is `data/overleaf_sync_20261110.json`. This release updates the
local source and GitHub; the Overleaf project itself has not been overwritten.


Upload `SE_JEPA_CVPR2027_Overleaf.zip` with **New Project → Upload Project**.
Select `main.tex` and the **pdfLaTeX** compiler. The project uses BibTeX and the
unchanged official CVPR 2027 style in anonymous review mode.

`main.tex` contains the paper, AI-use statement, references, and appendices A–E.
The appendix starts on a new page for reviewing the complete manuscript.
`supplement.tex` also builds the appendix separately; compile `main.tex` first
so its cross-reference file is available. No shell escape, Python, external
image links, SVG conversion, or custom fonts are required by Overleaf.

## Source organization

- `sec/`: abstract, main text, AI-use statement, and appendices. Section 3.1 motivates cross-episode replay, with the paired weight-inheritance diagnostic in main Table 2. Section 4.1 describes the evaluation setup; Section 4.6 compares the read and write protections and tests how to carry experience between episodes. The main text includes the same-checkpoint component comparison. Main Table 3 reports the same-checkpoint components and fixed slow learning rates. Appendix D gives the familiarity ablation and protocol details.
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
python tools/combine_teaser.py --template figures/source/teaser_combined.svg --bars-dir figures --output-dir figures --upper-pdf figures/source/teaser_before_alignment.pdf
cp figures/teaser_combined.svg figures/source/teaser_combined.svg
```

The combined exporter uses PyMuPDF and preserves the native upper panel; CairoSVG is only needed when no native upper PDF is supplied. `tools/build_condition_gains.py`
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
12.99 replans). The DreamerV3 and IRIS PushObj columns use the seven-shape CEM evaluation
at commit 31f9b34, independently checked from all 700 episodes. Zhou et al.
supply the rope and granular CDs for DreamerV3, IRIS, R3M, and DINO CLS. The two local frozen
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
rope example 1, and granular example 2. The two additional deformable cases
have been removed from the manuscript. All four retained panels come from native
Drive PDFs, preserving the original
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
Appendix A reports the server with eight NVIDIA A100 GPUs used for training
and evaluation.

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

The red-object row uses an exactly matched target shared by DINO-WM GD
(sample 31) and mixed14 (sample 30). Frozen and AdaJEPA use the official shape
checkpoint; SE-JEPA is round five of chain two. DINO-WM now uses the genuine
30-MPC rerun from the original epoch-three checkpoint. All three baselines fail
at 30 replans; SE-JEPA succeeds at six. The 460 delivery-file hashes, 151 frames,
30 feedback boundaries, seed mapping and original success criterion were checked.
`data/revision_assets/dinowm_visual_30mpc/` contains selected raw assets, the local
audit and the verbatim server delivery ZIP. The prior six-replan evidence stays
archived separately and is no longer used in Figure 3.

## Native observations in the mixed-domain figure
Figure 8 places 14 native observations in one row below the three learning panels.
Every thumbnail is episode zero from GD chain two, round five. They are extracted
from the saved initial observations, including actual blur, noise and darkness;
no new image corruption or simulation is applied. The images illustrate the stream
domains rather than additional performance claims. Their hashes and exact sample
identities are in `data/revision_assets/mixed14_observations/manifest.json`.

## PushT DINO-WM CEM update
Only PushT is changed: 99/300 shifted goals (33.0%), 56/150 recolored goals
(37.3%), and 22.05 five-action units (4.41 native MPC replans × 5). The checkpoint is the new random-data
epoch-three T-shape model. `tools/dinowm_cem.py` verifies the pinned final pipeline
summary against five archived raw condition files; red-object and red-goal use
the final summary because their raw CEM files are absent from that snapshot.
Full configurations remain in the data archive. Appendix B briefly explains the conversion; the main table labels Steps
as five-action units. PushObj, GD charts,
and the deformable benchmarks are unaffected.

Related-work catalogs and baseline descriptions have been condensed to focus
space on SE-JEPA, its persistent mixed-domain memory and the main ablations.

## Updated author abstract
`data/drive_reference/abstract1.md` is the current author-supplied abstract.
The manuscript reproduces it verbatim apart from TeX escaping. The earlier
`abstract_supplied.md` remains as a historical source. Numerical checks confirm
mean cosine 0.018, matched-goal success 62% to 48%, paired held-out gains of
15.41–18.74 percentage points and maximum CD reduction 48.58%. These round to
the supplied claims; no wording or numbers were changed. See
`data/abstract_results_audit.json` for exact values and source checksums.


## 2026-11-10 GD ablation supplement
Table 3 now covers PushObj chain 1 and PushT chains 1/2, including retention,
maturity/familiarity gates and fixed slow learning rates. Appendix D adds per-chain
success/replans and three-chain inheritance comparisons. All 51 GD conditions
(53,550 evaluations, including repeated test targets) were recomputed from retained
episode CSVs; 966 available raw NPZ batches were checked against the source hashes
and CSV records. Visual initial/goal states agree in all 21 seed/domain groups.
Historical PushObj full-model CSVs lack original state tensors; this audit limit is
stated in Appendix D. The shared visual episodic-reset evaluation is not counted
as an independent repeat for each learning chain. Data and audit:
`data/gd_ablation_20261110/`; reproduction: `tools/build_gd_ablation.py`.

The default remains the original 1e-4 slow learning rate and maturity gate. The
main text discusses the observed gate trade-off; future work proposes calibrating
eligibility to uncertainty and planning outcomes. The main paper occupies 8 pages,
with about 2.4 body-text lines left below the final paragraph. The complete PDF is
20 pages. Official CVPR style, geometry and fonts are unchanged.

The 1e-2 slow-rate row and its discussion have been removed from the displayed
ablation tables; the sweep now ends at 5e-4. The original evidence remains
archived. DreamerV3/IRIS PushObj columns are now complete from the new seven-shape
CEM evaluation. The earlier missing-cell audit is historical; current verification
is in `data/dreamer_iris_pushobj_20261110/local_audit.json`.

## Figure alignment and paragraph-ending review
Figure 1 expands its lower chart leftward while preserving the upper schematic
pixel for pixel. Figure 2 replaces the Square/T test-time thumbnails with original
Rope/Granular simulator observations from the existing deformable archive.
`tools/update_overview_tasks.py` updates the native PDF and editable SVG; source
image checksums are in `data/revision_assets/architecture_tasks/provenance.json`.

The rendered main PDF passes the 40%-width ending check for 120 prose paragraphs
and captions plus all 42 bibliography entries. Minor prose edits remove short
endings; bibliography line breaking is set locally without changing entry text.
Headings, displayed equations, algorithm steps and table cells are excluded.
The author abstract, numerical results, font sizes and page geometry are unchanged.
Recheck after changing the compiler or text with
`python tools/audit_paragraph_endings.py main.pdf main.aux report.json`.
The eight-page main text leaves 2.43 lines beneath the final paragraph.

## DreamerV3/IRIS completion and readability revision
The main table reports DreamerV3 PushObj 7.1% / 5.3% / 28.36 and IRIS
2.9% / 2.7% / 29.37 (All / Unseen / Steps). All 700 final-state outcomes,
action counts, matching targets and 78 source-file hashes were checked.
The Unseen group is I, square and small tee for both methods. Appendix C
briefly describes T-only pretraining and frozen offline CEM adapters. Existing
PushT and deformable results are unchanged; no additional table markers appear.
`tools/dreamer_iris_pushobj.py` verifies the archived values used by the table
builder. The pinned source branch is `codex/results-dreamerv3-iris-pushobj-20261110`.

The introduction now develops the recurring-shift problem, episodic relearning,
failed weight inheritance, trajectory consolidation, validated reuse and the
supporting experiments. All sections were reviewed for readable professional
prose; the supplied abstract is unchanged. Primary-paper reading notes are in
`data/readability_review_20261110.json`. The main text has eight pages with
2.43 trailing lines; all audited paragraph endings meet the 40% requirement.
