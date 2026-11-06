# SE-JEPA manuscript

This directory contains the CVPR-format review manuscript and its Overleaf source.

- [Combined paper and appendix (PDF)](main.pdf)
- [Overleaf upload archive](SE_JEPA_CVPR2027_Overleaf.zip)
- [Figure 1 combined editable SVG, original artwork, chart data and rebuilding scripts](SE_JEPA_Figure1_Source.zip)
- [Editable source](SE_JEPA_CVPR2027_Overleaf/main.tex)
- [Build and data notes](SE_JEPA_CVPR2027_Overleaf/README_OVERLEAF.md)

The appendix starts on a separate page after the references. The source includes
vector figures, selected original simulator frames, numerical snapshots, and
scripts for rebuilding the figures. Compile `main.tex` with pdfLaTeX and BibTeX.

Section 3.1 motivates cross-episode replay using the paired weight-inheritance experiment. Appendix D gives the full diagnostic. Section 4.1 introduces the evaluation protocols. The main table reports the seven-shape PushObj protocol with the original unseen subset; Appendix C lists the separate two-shape literature results. Section 4.5 studies removal of the familiarity checks for both memory reads and writes. The single-column Figure 1 pairs the recurring-shift schematic with held-out planning success. The marked entries in Figure 1 (GD) and the main table (CEM) show this variant on all seven shapes. All familiarity variants retain the maturity screen. Section 4.5 compares replay with weight inheritance under a common initial checkpoint. Appendix D gives the slow learning rate sweep. The [experiment archive](https://github.com/Celveen/SE-JEPA/tree/3faad041417c6f2a22df2ad62822c34fd075c4bb/experiments/ablation_archive_20261106) contains the underlying runs.

The revised layout places the main table on page 5 and all main result figures
before Conclusion on page 8. Figure 7 shows the goal-success gains for each shape and visual shift. The bibliography contains 42 cited references;
Appendix A records the author-supplied 8-GPU RTX 5090 and A100 compute systems.

The backbone discussion distinguishes task-trained ResNet models on PushObj/PushT
from DINO-WM on rope/granular. Main Figure 6 presents five-round unseen-shape
and color-shift success together with four-pass rope/granular CD. Its archived
counts and uncertainty definitions are in `data/learning_progress.json` within
the source project. Future work includes improving planning as prediction
accuracy and behavioral performance become decoupled later in deployment.
