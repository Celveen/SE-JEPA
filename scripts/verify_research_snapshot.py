import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    arguments = parser.parse_args()
    root = arguments.snapshot.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Path escapes snapshot: {entry['path']}")
        stored = path.read_bytes()
        assert len(stored) == entry["bytes"], entry["path"]
        assert hashlib.sha256(stored).hexdigest() == entry["sha256"], entry["path"]
        payload = gzip.decompress(stored) if entry["gzip"] else stored
        assert len(payload) == entry["source_bytes"], entry["path"]
        assert hashlib.sha256(payload).hexdigest() == entry["source_sha256"], entry["path"]
    assert len(manifest["runs"]) == manifest["completed_directories"]
    print(f"Verified {len(manifest['files'])} payloads across {manifest['completed_directories']} completed evaluation directories.")


if __name__ == "__main__":
    main()
