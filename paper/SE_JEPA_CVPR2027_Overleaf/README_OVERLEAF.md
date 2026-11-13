# SE-JEPA — CVPR 2027 manuscript

## Current revision: 2026-11-13

The latest Overleaf teacher edits were downloaded before editing. The teacher's new abstract wording and Related Work are preserved. Repeated motivation and result explanations were shortened to keep the main paper at eight pages.

Figure 1 now uses the author's `teaser_final(6).pdf`, with empty margins cropped and vector/text content retained. It spans both columns on page 2. Main Table 5 tests mature-predictor selection on future observations. The cost result is summarized in main Section 4.2, with detailed latency, GPU memory, phase breakdown, and prior consolidation/storage tables in Appendix A.4. Appendix D.11 documents the routing diagnostic and paired uncertainty.

## Build and synchronization

Use `main.tex` with pdfLaTeX on Overleaf. `supplement.tex` builds the standalone appendix after `main.aux` exists. The official CVPR 2027 style, font size, margins, and line spacing are unchanged. The combined manuscript has 22 pages: main 8, statement/references 2, appendix 12; the standalone supplement has 13 pages.

Overleaf was updated selectively: 14 files were verified after upload and the other 45 pre-existing files were unchanged. The online compile also has eight main pages and the teaser on page 2. Local previews use Tectonic/XeTeX. Its 146 prose/caption endings and 42 bibliography endings pass the 40% width check, and the final main-page column gap is 2.62 lines. The standalone appendix passes its corresponding checks. pdfLaTeX may choose different line breaks.

The 53-file clean archive contains only active compile dependencies and its README. It excludes obsolete illustrations, unused tables, raw data, and tools. The full research archive retains source art, historical records, audited numeric data, and reproduction tools.

## Evidence

- `data/revision_20261113/`: verified routing and cost summaries, paired task data, storage inventory, and Overleaf verification.
- `tools/build_routing_cost_tables.py`: rebuild the four new tables.
- `data/cem_ablation_20261112/` and `tools/build_planner_ablation.py`: prior GD/CEM ablations.
- `data/revision_assets/architecture_tasks/text_layer_audit.json`: selectable architecture labels.
- `data/paragraph_layout_audit.json` and `compile_checks.json`: current build/layout checks.

Dated review-response files and author notes describe historical revisions; this section records the current state.
