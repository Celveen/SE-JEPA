# SE-JEPA — CVPR 2027 manuscript

## Current revision: 2026-11-12

The accepted abstract is unchanged. The main paper remains eight pages, with
20 pages including references and appendices. The main ablation table now shows
one complete GD/CEM stream for each task (PushObj chain 1, PushT chain 2).
The appendix separates the available GD memory-design comparisons from the
complete four-column GD/CEM gate and learning-rate grid. Incomplete PushT chain
columns and the redundant familiarity/three-chain displays have been removed.

Teacher-review changes clarify write validation versus retrieval, the 1.2 write
threshold and 0.9 maturity criterion, and buffer/predictor handling at a merge.
Repeated checkpoint and rerun commentary is removed; evaluation phases, sample
sizes, initialization facts and published-score sources remain explicit.
Figure 1 uses held-out GD results; Table 1 uses online CEM results for SE-JEPA
PushObj. These are distinct measurements, not contradictory averages.

Figure 2 retains the author's supplied colors, positions and fonts, plus real
rope/granular simulator images. Its original vector outlines now have an aligned
searchable text layer using the original embedded fonts. All 59 labels can be
copied, and the rendered pixels are unchanged. Rebuild with
`python tools/update_overview_tasks.py`.

## Build and upload

Use `main.tex` with pdfLaTeX on Overleaf; no shell escape or figure regeneration
is required. `supplement.tex` builds the standalone appendix after `main.aux`
exists. The official CVPR 2027 style is unchanged. The supplied preview is
validated locally with Tectonic/XeTeX; pdfLaTeX may choose slightly different
line breaks. The clean 49-file archive contains only active compile dependencies.
The current revision has not automatically overwritten the existing online project.

## Evidence and reproduction

- `data/review_response_20261112.json`: review-to-change mapping and prioritized missing experiments.
- `data/cem_ablation_20261112/`: selected episode records, source hashes and recomputed metrics.
- `tools/build_planner_ablation.py`: regenerate the new tables from GD/CEM episode outcomes.
- `data/revision_assets/architecture_tasks/text_layer_audit.json`: font and visual-preservation checks.
- `data/paragraph_layout_audit.json` and `compile_checks.json`: current layout/build checks.

All 21,000 CEM evaluation records were recomputed against the supplied summary;
399 original NPZ batches (19,950 evaluations) were independently checked.
The retained common grid contains 14,700 CEM evaluations. Three test seeds are
evaluation repetitions, not independent training runs. Missing CEM memory-design
ablations are not filled with other planners' results.

The research archive retains historical source evidence for reproducibility;
only the active tables and figures are included in the clean Overleaf archive.
