"""Update Figure 1's lower panel and export the complete editable SVG and vector PDF."""
import argparse
from copy import deepcopy
from pathlib import Path
import re
from xml.etree import ElementTree as ET

import fitz


NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def combine(template, bars_dir, output_dir, upper_pdf_path=None):
    svg = ET.parse(template)
    root = svg.getroot()
    panel = root.find(f"{{{NS}}}g[@id='held-out-planning-success']")
    if panel is None:
        raise ValueError("The template needs a lower 'held-out-planning-success' group.")
    translation = re.fullmatch(r"translate\(0[ ,]+([\d.]+)\)", panel.get("transform", ""))
    if translation is None:
        raise ValueError("The lower panel must be positioned with translate(0 y).")
    chart_y = float(translation.group(1))
    width = float(root.get("viewBox").split()[2])
    bars = ET.parse(bars_dir / "teaser_bars.svg").getroot()
    x, y, chart_width, chart_height = map(float, bars.get("viewBox").split())
    if (x, y) != (0, 0) or abs(chart_width - width) > .01:
        raise ValueError("The chart must use the same width and a zero-origin viewBox.")
    height = chart_y + chart_height
    root.set("height", f"{height:.5f}pt")
    root.set("viewBox", f"0 0 {width:g} {height:.5f}")
    for node in list(panel):
        panel.remove(node)
    for node in bars:
        if node.tag != f"{{{NS}}}metadata":
            panel.append(deepcopy(node))

    # The separate native chart PDF carries the embedded fonts used by the plotting script.
    upper = deepcopy(root)
    upper.remove(upper.find(f"{{{NS}}}g[@id='held-out-planning-success']"))
    upper.set("height", f"{chart_y:.5f}pt")
    upper.set("viewBox", f"0 0 {width:g} {chart_y:.5f}")
    if upper_pdf_path is None:
        import cairosvg
        upper_pdf = cairosvg.svg2pdf(bytestring=ET.tostring(upper, encoding="utf-8"))
    else:
        upper_pdf = Path(upper_pdf_path).read_bytes()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "teaser_combined.pdf"
    with fitz.open("pdf", upper_pdf) as top, fitz.open(bars_dir / "teaser_bars.pdf") as bottom, fitz.open() as doc:
        if len(bottom) != 1 or abs(bottom[0].rect.width-width) > .01 or abs(bottom[0].rect.height-chart_height) > .01:
            raise ValueError("The chart PDF and SVG must have the same dimensions.")
        page = doc.new_page(width=width, height=height)
        page.show_pdf_page(fitz.Rect(0, 0, width, chart_y), top, 0, clip=fitz.Rect(0, 0, width, chart_y))
        page.show_pdf_page(fitz.Rect(0, chart_y, width, height), bottom, 0)
        doc.set_metadata({"title": "SE-JEPA: recurring shifts and held-out planning"})
        pdf_path.write_bytes(doc.tobytes(garbage=4, deflate=True))
    svg.write(output_dir / "teaser_combined.svg", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path("teaser_combined.svg"))
    parser.add_argument("--bars-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("rebuilt"))
    parser.add_argument("--upper-pdf", type=Path, help="Preserve the native upper panel from an existing combined PDF.")
    args = parser.parse_args()
    combine(args.template, args.bars_dir, args.output_dir, args.upper_pdf)
