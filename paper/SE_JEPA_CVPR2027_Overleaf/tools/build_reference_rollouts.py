#!/usr/bin/env python3
"""Retain the four requested reference layouts and two additional native cases.

This script only crops native PDF pages, or copies already complete pages.
It never rasterizes, synthesizes, reorders, recolors, or relabels figure content.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pymupdf as fitz


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
SOURCE = FIG / "source"
MANIFEST = ROOT / "data/revision_assets/reference_rollout_sources.json"
SOURCE_IDS = {
    "rope": "1y8s0Bc8ovsgEmXGsGYa8I48G6SaP7w5r",
    "granular": "1_15c-f8csFOvFTn4VGdgB_MnSk1ZoTZV",
    "square": "11HLsl5oLcs6P7lhTke_EWBRYU8hU3LPe",
    "redagent": "1IkMaFWyJLd0OT93J7dCT0Uxgez0xlUCX",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_records(page, clip):
    records = []
    for item in page.get_image_info(hashes=True):
        rect = fitz.Rect(item["bbox"])
        if not clip.contains(rect):
            continue
        records.append({
            "pixel_md5": item["digest"].hex(),
            "width": item["width"],
            "height": item["height"],
            "bbox_relative_to_clip": [
                round(rect.x0 - clip.x0, 3), round(rect.y0 - clip.y0, 3),
                round(rect.x1 - clip.x0, 3), round(rect.y1 - clip.y0, 3),
            ],
        })
    return records


def build(name, task, example=None):
    native = SOURCE / (
        f"{task}_full_original.pdf" if example else f"{task}_reference_original.pdf"
    )
    target = FIG / f"{name}.pdf"
    original = fitz.open(native)
    page = original[0]
    if example == 1:
        clip = fitz.Rect(0, 0, page.rect.width, 184.359)
    elif example == 2:
        clip = fitz.Rect(0, 190.944, page.rect.width, page.rect.height)
    else:
        clip = page.rect
    before_text = page.get_text(clip=clip)
    before_images = image_records(page, clip)

    if example:
        output = fitz.open()
        out_page = output.new_page(width=clip.width, height=clip.height)
        out_page.show_pdf_page(out_page.rect, original, 0, clip=clip)
        output.save(target, garbage=4, deflate=True)
        output.close()
    else:
        # These native pages contain no screenshot border or surrounding UI.
        shutil.copyfile(native, target)

    result = fitz.open(target)
    out_page = result[0]
    after_text = out_page.get_text(clip=out_page.rect)
    after_images = image_records(out_page, out_page.rect)
    assert after_text == before_text, f"Text was changed or clipped in {name}"
    assert after_images == before_images, f"Image pixels/placement changed in {name}"
    if example:
        assert len(after_images) == 21, f"Expected all 3 x 7 frames in {name}"
    else:
        assert "raw loss levels are not directly comparable" in after_text
        assert "Gray frames hold the terminal state" in after_text
        assert sha256(native) == sha256(target)

    record = {
        "figure": str(target.relative_to(ROOT)),
        "task": task,
        "example": example,
        "source_file": str(native.relative_to(ROOT)),
        "drive_file_id": SOURCE_IDS[task],
        "drive_url": f"https://drive.google.com/file/d/{SOURCE_IDS[task]}/view",
        "source_sha256": sha256(native),
        "output_sha256": sha256(target),
        "source_dimensions_pt": [page.rect.width, page.rect.height],
        "crop_rect_pt": list(clip),
        "output_dimensions_pt": [out_page.rect.width, out_page.rect.height],
        "operation": "native PDF crop" if example else "byte-identical native PDF copy",
        "visible_embedded_image_placements": len(after_images),
        "all_visible_image_pixels_and_positions_preserved": True,
        "all_visible_text_preserved": True,
        "visible_image_records": after_images,
    }
    if task == "redagent":
        record["reference_difference"] = (
            "The archived native PDF correctly labels the displayed trajectories as Replay; "
            "the user screenshot has Simulator labels and truncates the lower explanation. "
            "The native Replay labels and full explanation are preserved."
        )
    result.close()
    original.close()
    return record


def main():
    cases = [
        ("reference_rope", "rope", 1),
        ("reference_granular", "granular", 2),
        ("reference_square", "square", None),
        ("reference_redagent", "redagent", None),
        ("additional_rope", "rope", 2),
        ("additional_granular", "granular", 1),
    ]
    manifest = {
        "purpose": "Preserve all four user reference layouts and two additional deformable cases.",
        "source_policy": "All six outputs use original Drive PDFs, with native vectors and image pixels preserved.",
        "content_edits": "None. Original labels, explanatory notes, curves, final CD values, and success markers are retained.",
        "deformable_provenance": {
            "predictor": "Third consolidation pass, slow-only cl_noada_r3; main quantitative table uses the fourth pass.",
            "selection": "Cases selected from the repeated 50-instance stream where both baselines fail and ours succeeds; not held-out test instances.",
            "dino_wm_frames": "Recorded actions replayed in the original FleX simulator.",
            "adajepa_and_ours_frames": "Simulator frames saved during planning.",
            "final_cd": "Values recorded by original runtime evaluation, not remeasured from replay.",
        },
        "rigid_provenance": {
            "planner": "CEM",
            "sample": "First held-out target, test seed 2000.",
            "square": "Frozen trajectory is action replay; AdaJEPA and Ours are simulator frames saved during execution.",
            "redagent": "All displayed trajectories are action replay in the original simulator.",
            "prediction_losses": "Each method is scored with its own model weights. Absolute loss levels are not directly comparable across methods.",
        },
        "figures": [build(*case) for case in cases],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    for item in manifest["figures"]:
        print(item["figure"], item["output_dimensions_pt"], item["output_sha256"])


if __name__ == "__main__":
    main()
