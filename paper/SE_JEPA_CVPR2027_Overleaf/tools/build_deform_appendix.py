"""Relabel archived real deformable trajectories without altering simulator frames."""
from pathlib import Path
import hashlib
import json

from matplotlib import font_manager
import pymupdf as fitz


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "rope": "1y8s0Bc8ovsgEmXGsGYa8I48G6SaP7w5r",
    "granular": "1_15c-f8csFOvFTn4VGdgB_MnSk1ZoTZV",
}


def image_signature(page):
    """Decoded image hashes and placements, including repeated start/goal frames."""
    return [
        {"bbox": list(info["bbox"]), "pixels_md5": info["digest"].hex()}
        for info in page.get_image_info(hashes=True)
    ]


def prepare(task, file_id):
    source = ROOT / "figures/source" / f"{task}_full_original.pdf"
    output = ROOT / "figures" / f"{task}_full.pdf"
    fontfile = str(font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    ))
    font = fitz.Font(fontfile=fontfile)
    with fitz.open(source) as doc:
        assert len(doc) == 1
        page = doc[0]
        before_images = image_signature(page)
        before_text = page.get_text()
        assert len(before_images) == 42
        replacements = []
        blocks = page.get_text(
            "dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES
        )["blocks"]
        for block in blocks:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"] != "Ours":
                        continue
                    assert line["dir"] == (0.0, -1.0)
                    replacements.append(span)
                    page.add_redact_annot(span["bbox"], fill=False, cross_out=False)
        assert len(replacements) == 2
        page.apply_redactions(images=0, graphics=0)
        page.insert_font(fontname="sejepa_label", fontfile=fontfile)
        for span in replacements:
            center_y = (span["bbox"][1] + span["bbox"][3]) / 2
            length = font.text_length("SE-JEPA", fontsize=span["size"])
            page.insert_text(
                (span["origin"][0], center_y + length / 2),
                "SE-JEPA", fontsize=span["size"], fontname="sejepa_label",
                color=fitz.sRGB_to_pdf(span["color"]), rotate=90,
            )
        doc.subset_fonts()
        doc.save(output, garbage=4, deflate=True)
    with fitz.open(output) as doc:
        page = doc[0]
        assert image_signature(page) == before_images
        after_text = page.get_text()
        assert after_text.count("SE-JEPA") == 2 and "Ours" not in after_text
        # Redaction appends new text objects; compare the remaining original words.
        assert before_text.replace("Ours", "").split() == after_text.replace("SE-JEPA", "").split()
        return {
            "drive_id": file_id,
            "drive_url": f"https://drive.google.com/file/d/{file_id}/view",
            "source": str(source.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output": str(output.relative_to(ROOT)),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "page_size_pt": list(page.rect),
            "cases": 2,
            "image_placements": len(before_images),
            "decoded_image_hashes_and_positions_unchanged": True,
            "images": before_images,
        }


def main():
    manifest = {
        "source": "Archived Drive native PDF exports of real FleX trajectories",
        "edit": "Only replace the two vertical Ours labels per page with SE-JEPA",
        "columns": ["start", "push 3", "push 6", "push 9", "push 12", "push 15", "goal"],
        "selection": "Two examples per material where both baselines fail and SE-JEPA succeeds",
        "state": "Third consolidation pass, slow-only; main table uses fourth-pass aggregate",
        "frame_origin": "Frozen DINO-WM executed-action replay; AdaJEPA/SE-JEPA saved planning frames",
        "cd_origin": "Runtime final CD annotations retained unchanged",
        "protocol": "data/drive_reference/deform_readme.txt",
        "files": {task: prepare(task, file_id) for task, file_id in SOURCES.items()},
    }
    destination = ROOT / "data/revision_assets/deform_rollout_sources.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
