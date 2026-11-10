# SE-JEPA manuscript

This directory contains the CVPR-format review manuscript and its Overleaf source.

- [Combined paper and appendix (PDF)](main.pdf)
- [Clean Overleaf upload archive (48 files)](SE_JEPA_CVPR2027_Overleaf_Clean.zip)
- [Full research source archive with data and scripts](SE_JEPA_CVPR2027_Overleaf.zip)
- [Figure 1 combined editable SVG, original artwork, chart data and rebuilding scripts](SE_JEPA_Figure1_Source.zip)
- [Editable source](SE_JEPA_CVPR2027_Overleaf/main.tex)
- [Build and data notes](SE_JEPA_CVPR2027_Overleaf/README_OVERLEAF.md)

The appendix starts on a separate page after the references. The source includes
vector figures, selected original simulator frames, numerical snapshots, and
scripts for rebuilding the figures. Compile `main.tex` with pdfLaTeX and BibTeX.

Section 3.1 motivates cross-episode replay using the paired weight-inheritance experiment. Main Table 2 gives the matched-goal accumulation diagnostic. Section 4.1 and Appendix C describe the evaluation protocols and identify the published PushObj Unseen and deformable results in the main table. Figure 1 pairs the recurring-shift schematic with held-out GD success. Figure 1 uses the three-chain GD mean for SE-JEPA. The main-table SE-JEPA PushObj entries use fifth-round online CEM outcomes, with the selected chains documented in Appendix B. Section 4.6 studies removal of the familiarity checks and compares replay with weight inheritance under a common initial checkpoint. Main Table 3 includes the fixed slow learning rate sweep. [Ablation archive](https://github.com/Celveen/SE-JEPA/tree/3faad041417c6f2a22df2ad62822c34fd075c4bb/experiments/ablation_archive_20261106)

The revised layout places the main table on page 5 and all main result figures
before Conclusion on page 8. Figure 7 shows the goal-success gains for each shape and visual shift in a
single-column, vertically stacked layout. The appendix adds two complete
rope/granular rollout examples from the original Drive PDFs; the combined
manuscript has 20 pages, including eight main-text pages. The bibliography contains 42 cited references;
Appendix A records the author-supplied 8-GPU RTX 5090 and A100 compute systems.

The backbone discussion distinguishes task-trained ResNet models on PushObj/PushT
from DINO-WM on rope/granular. Main Figure 6 presents five-round unseen-shape
and color-shift success together with four-pass rope/granular CD. Its archived
counts and uncertainty definitions are in `data/learning_progress.json` within
the source project. Future work includes improving planning as prediction
accuracy and behavioral performance become decoupled later in deployment.

The mixed 14-domain experiment has a dedicated main-text Section 4.5 on page 7
and a three-panel Figure 8 on page 8, covering overall, shape and visual
five-round learning. Figure 8 averages GD and CEM equally and adds Frozen
and AdaJEPA as horizontal references. Overall SE-JEPA success increases from
65.3% to 75.5%; the detailed tables retain the individual planner results. Appendix D retains Tables 8–9 and the detailed protocol, including
repeated targets and the baseline initialization differences.
Appendix E includes all four original reference panels and their explanatory
text. The final extra deformable figure has been removed at the author’s request. The original loss curves and
notes are preserved. The clean Overleaf archive contains 48 files and 13 figures.

The main-table PushObj CEM row is 80.4% All, 73.1% Unseen, and 12.99 replans,
from the selected three fifth-round online streams. Paired post-learning
evaluations remain separately reported in Appendix B.

Figure 3 uses the verified real DINO-WM visual rerun, with matched starts and
goals. All three baselines fail at 30 MPC replans and SE-JEPA succeeds at six.
The original server delivery and an independent local evidence audit are in
`data/revision_assets/dinowm_visual_30mpc/`. The original handoff package remains
archived for provenance.

Figure 8 adds a single row of 14 native simulator observations, one per domain,
below its three learning panels. Frames use episode zero of GD chain two, round
five, without outcome-based selection or image alteration.

Only the DINO-WM PushT CEM cells have been replaced by the random-data epoch-three
checkpoint: Shift 33.0%, Color 37.3%, Steps 22.05 in five-action units. PushObj is unchanged.
`data/dinowm_pusht_cem_epoch3/` archives the pinned pipeline summary and five
available raw episode files; the red-object/red-goal values are summary-backed.
Baseline descriptions have been shortened while retaining checkpoint provenance,
references and necessary evaluation distinctions.

The abstract now uses the author’s `abstract1.md` verbatim. Its quantitative
claims were checked against the paired held-out comparisons, accumulation
diagnostic and fourth-pass deformable results; see `data/abstract_results_audit.json`.
Appendix B briefly explains DINO-WM PushT’s Steps conversion (4.41 × 5 = 22.05).


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
with about 1.5 body-text lines left below the final paragraph. The complete PDF is
20 pages. Official CVPR style, geometry and fonts are unchanged.

The 1e-2 slow-rate row and its discussion have been removed from the displayed
ablation tables; the sweep now ends at 5e-4. The original evidence remains
archived. DreamerV3/IRIS PushObj All and Steps are still unavailable for the
seven-shape protocol; Appendix C explains these dashes. See
`data/dreamer_iris_missing_cells_audit.json` for the current source review.

Figure 1's lower GD chart now extends leftward. Figure 2 includes original rope
and granular robot-arm observations in its test-time task examples. The full
manuscript review verifies at least 40% final-line width for prose, captions and
bibliography entries; the original abstract and numerical results are unchanged.
The main text remains eight pages, with 1.43 lines of trailing space.
