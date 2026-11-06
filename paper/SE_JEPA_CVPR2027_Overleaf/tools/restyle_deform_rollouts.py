"""Typeset the method labels on the archived Drive figure, keeping its FleX panels."""
from matplotlib import font_manager


def draw_deform_rollouts(source, output):
    import fitz

    with fitz.open(source) as doc:
        page = doc[0]
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]
        spans = [s for b in blocks for line in b["lines"] for s in line["spans"]]
        frames = [fitz.Rect(im["bbox"]) for im in page.get_image_info()]
        labels = [s for s in spans if s["text"].startswith("(")]
        scores = [s for s in spans if s["text"].startswith("CD ")]
        if not len(labels) == len(scores) == len(frames) == 16:
            raise ValueError("Expected sixteen labelled FleX panels from the archived Drive figure")

        replacements = []
        outlines = []
        for label, score, frame in zip(labels, scores, frames):
            center = (frame.x0 + frame.x1) / 2
            ours = label["text"].endswith(" Ours")
            for span in (label, score):
                box = fitz.Rect(span["bbox"])
                if not (box.x0 <= center <= box.x1 and frame.y1 < box.y0 < frame.y1 + 12):
                    raise ValueError("The label and CD must align with their original simulator frame")
                page.add_redact_annot(box, fill=False, cross_out=False)
                replacements.append((span, center, "bold" if ours else "normal"))
            if ours:
                outlines.append(frame)
        if len(outlines) != 4:
            raise ValueError("Expected one SE-JEPA outcome for each of four instances")
        page.apply_redactions(images=0, graphics=0)
        fonts = {}
        for weight in ("normal", "bold"):
            path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight))
            fonts[weight] = fitz.Font(fontfile=path)
            page.insert_font(fontname=weight, fontfile=path)
        for span, center, weight in replacements:
            text = span["text"].replace(" Ours", " SE-JEPA")
            size = span["size"]
            x = center - fonts[weight].text_length(text, fontsize=size) / 2
            color = fitz.sRGB_to_pdf(span["color"])
            page.insert_text((x, span["origin"][1]), text, fontname=weight, fontsize=size, color=color)
        for frame in outlines:
            page.draw_rect(frame, color=(.149, .208, .282), width=1.35)
        doc.subset_fonts()
        doc.save(output, garbage=4, deflate=True)
