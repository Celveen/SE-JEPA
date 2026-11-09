"""Recheck the archived fifth-round CEM outcomes used by the main table."""
from pathlib import Path
import hashlib
import json
import numpy as np


def verified_online_results(directory):
    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text())
    rows = audit["raw_rows"]
    assert len(rows) == 21
    for row in rows:
        source = row["source"]
        path = directory / source["path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source["sha256"], path
        blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
        assert hashlib.sha1(blob).hexdigest() == source["git_blob_sha"], path
        with np.load(path, allow_pickle=False) as data:
            success = data["successes"].astype(bool)
            steps = np.where(success, data["action_len"], 30)
            assert len(success) == row["episodes"] == 50
            assert int(success.sum()) == row["successes"]
            assert float(steps.sum()) == row["replans_sum"]
    for group, summary in audit["summary"].items():
        selected = [r for r in rows if group == "all" or r["group"] == group]
        n = sum(r["episodes"] for r in selected)
        k = sum(r["successes"] for r in selected)
        steps = sum(r["replans_sum"] for r in selected)
        assert n == summary["episodes"] and k == summary["successes"]
        assert abs(100 * k / n - summary["mean_pct"]) < 1e-10
        assert abs(steps / n - summary["mean_replans"]) < 1e-10
    return audit

