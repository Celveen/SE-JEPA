# SE-JEPA manuscript

This directory contains the CVPR-format review manuscript and its Overleaf source.

- [Combined paper and appendix (PDF)](main.pdf)
- [Clean Overleaf upload archive (49 files)](SE_JEPA_CVPR2027_Overleaf_Clean.zip)
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
single-column, vertically stacked layout. The appendix adds four complete
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
Appendix E includes all four original reference panels, two additional
deformable cases, and their explanatory text. The original loss curves and
notes are preserved. The clean Overleaf archive contains 49 files and 15 figures.

The main-table PushObj CEM row is 80.4% All, 73.1% Unseen, and 12.99 replans,
from the selected three fifth-round online streams. Paired post-learning
evaluations remain separately reported in Appendix B.

Figure 3 is slightly smaller and includes DINO-WM GD in both rows, with exact
initial/goal matching across all four methods. The I-shape case uses the paired
shape evaluation; the red-object case comes from the mixed 14-domain stream.
Labels count executed environment steps because the visual DINO-WM run uses
a different replanning frequency. The original red-object case remains archived.
The Goal explanatory line has been removed.
