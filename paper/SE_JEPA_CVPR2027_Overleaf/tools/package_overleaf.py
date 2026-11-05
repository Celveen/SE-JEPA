"""Create a clean, self-contained Overleaf upload archive."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
files = [root / name for name in (
    "main.tex", "supplement.tex", "preamble.tex", "references.bib",
    "cvpr.sty", "ieeenat_fullname.bst", ".latexmkrc",
    "README_OVERLEAF.md", "AUTHOR_NOTES.md",
)]
for directory, suffixes in [
    ("sec", {".tex"}), ("tables", {".tex"}),
    ("figures", {".pdf", ".png", ".svg"}),
    ("data", {".json", ".npy", ".npz", ".md", ".csv", ".png", ".txt"}),
    ("tools", {".py"}),
]:
    files.extend(p for p in (root / directory).rglob("*") if p.is_file() and p.suffix in suffixes)
if (root / "compile_checks.json").exists():
    files.append(root / "compile_checks.json")
files = sorted(set(files))
assert all(p.is_file() for p in files)
assert hashlib.sha256((root / "cvpr.sty").read_bytes()).hexdigest() == "1cd7f434b13d41c475df4a1901a5d82d9ca8b80b8cf6205dc4602a5c164286b4"
manifest = {
    "created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
    "main_file": "main.tex", "supplement_file": "supplement.tex", "compiler": "pdfLaTeX",
    "official_cvpr_style_unchanged": True,
    "files": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
}
destination = root / "SE_JEPA_CVPR2027_Overleaf.zip"
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        relative = str(path.relative_to(root))
        if relative == "data/paper_snapshot.json":
            # Remove server-account paths from the portable numeric snapshot.
            value = path.read_text().replace(str(root.parent), "{WORKSPACE}")
            archive.writestr(relative, value)
            manifest["files"][relative] = hashlib.sha256(value.encode()).hexdigest()
        else:
            archive.write(path, relative)
    archive.writestr("build_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
print(destination)
print(f"{len(files)} source/assets files; {destination.stat().st_size / 1024 / 1024:.2f} MiB")
