# SE-JEPA manuscript

This directory contains the CVPR-format review manuscript and its Overleaf source.

- [Combined paper and appendix (PDF)](main.pdf)
- [Overleaf upload archive](SE_JEPA_CVPR2027_Overleaf.zip)
- [Editable source](SE_JEPA_CVPR2027_Overleaf/main.tex)
- [Build and data notes](SE_JEPA_CVPR2027_Overleaf/README_OVERLEAF.md)

The appendix starts on a separate page after the references. The source includes
vector figures, selected original simulator frames, numerical snapshots, and
scripts for rebuilding the figures. Compile `main.tex` with pdfLaTeX and BibTeX.

Section 3.1 motivates cross-episode replay using the paired weight-inheritance experiment. Appendix D gives the full diagnostic. Section 4.1 introduces the evaluation protocols. The main table separates the original two-shape PushObj results in Zhou et al. from the seven-shape evaluation in this study, and identifies the fixed baseline as Frozen (pretrained).

The revised layout places the main table on page 5 and all main result figures
before Conclusion on page 7. The bibliography contains 42 cited references;
Appendix A records the author-supplied 8-GPU RTX 5090 and A100 compute systems.

The backbone discussion distinguishes task-trained ResNet models on PushObj/PushT
from DINO-WM on rope/granular. Main Figure 7 presents five-round unseen-shape
and color-shift success together with four-pass rope/granular CD. Its archived
counts and uncertainty definitions are in `data/learning_progress.json` within
the source project. Future work includes improving planning as prediction
accuracy and behavioral performance become decoupled later in deployment.
