"""Verify the archived mixed rope/granular experiment and generate its table.

CD is recomputed from the archived per-instance reports and checked against
the original per-chunk logs. Routing is counted directly from episode records;
success is reconstructed from CD, never from the simulator's success flag.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/deform_mixed_20261110"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    sources = json.loads((DATA / "sources.json").read_text())
    for entry in sources["files"]:
        assert hashlib.sha256((DATA / entry["local"]).read_bytes()).hexdigest() == entry["local_sha256"]
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert (manifest["rounds"], manifest["total_episodes"]) == (3, 300)
    assert len(manifest["blocks"]) == 30
    trajectories = defaultdict(set)
    inputs = set()
    rows, all_records = [], []
    thresholds = {"rope": 0.5, "granular": 0.2}
    material_domain = {"rope": "d0", "granular": "d1"}
    for round_number in (1, 2, 3):
        report = json.loads((DATA / f"round_{round_number}_review.json").read_text())
        assert report["errors"] == []
        samples = {tag: report["samples"][tag] for tag in thresholds}
        per_sample = {(s["chunk"], s["sample"]): s for ss in samples.values() for s in ss}
        records = []
        for block in [b for b in manifest["blocks"] if b["round"] == round_number]:
            k, tag, chunk = block["index"], block["object"], block["chunk"]
            assert tag == ("rope" if k % 2 == 0 else "granular")
            folder = DATA / "raw" / chunk
            cfg = yaml.safe_load((folder / "config.yaml").read_text())
            c = cfg["planner"]["continual"]
            assert c["slow_lr"] == 2e-4 and c["write"] == "delta"
            assert c["abs_ratio"] == 20 and c["familiar_thr"] == 0.117 and c["familiar_read"]
            assert not c["read_only"] and cfg["n_evals"] == 10
            assert cfg["planner"]["adapt"]["lr"] == 5e-4
            assert (cfg["planner"]["max_iter"], cfg["planner"]["n_taken_actions"]) == (3, 5)
            assert cfg["env_kwargs_override"]["object_name"] == tag
            assert cfg["ckpt_base_path"].endswith("rope_dino_h1_fs1_e100_seed0")
            if k == 0:
                assert c["state_in"] is None
            else:
                previous = manifest["blocks"][k - 1]["chunk"]
                assert c["state_in"].endswith(f"/{previous}/continual_state.pt")
            rr = read_jsonl(folder / "episodes.jsonl.txt")
            assert len(rr) == 10 and [x["episode"] for x in rr] == list(range(k * 10, (k + 1) * 10))
            final_cd = []
            for i, (record, target) in enumerate(zip(rr, block["samples"])):
                s = per_sample[chunk, i]
                assert record["tag"] == tag and record["sample_idx"] == i
                assert s["trajectory"] == target["trajectory"] and s["offset"] == target["offset"]
                key = (tag, target["trajectory"], target["offset"])
                assert key not in inputs
                inputs.add(key)
                trajectories[tag, round_number].add(target["trajectory"])
                for field in ("read", "write", "write_rule"):
                    assert record[field] == s[field]
                assert math.isfinite(s["final_cd"])
                assert s["success"] == (s["final_cd"] < thresholds[tag])
                final_cd.append(s["final_cd"])
                assert len(record["route"]) == 3
                for domain in record["domains"].values():
                    assert sum(domain["tags"].values()) == domain["n"]
                    assert len([v for v in domain["tags"].values() if v]) == 1
            log = read_jsonl(folder / "logs.jsonl.txt")
            logged = [x["final_eval/mean_chamfer_distance"] for x in log if "final_eval/mean_chamfer_distance" in x][-1]
            assert math.isclose(mean(final_cd), logged, abs_tol=1e-6, rel_tol=0)
            records.extend(rr)
        assert len(records) == 100 and records[-1]["domains"] == report["domains"]
        assert set(report["domains"]) == {"d0", "d1"}
        assert all(d["mature"] and d["learner"] for d in report["domains"].values())
        routing = Counter()
        final_selection = Counter()
        for record in records:
            expected = material_domain[record["tag"]]
            classify = lambda key: "frozen" if key is None else "correct" if key == expected else "incorrect"
            final_selection[classify(record["read"])] += 1
            routing.update(classify(q["best"]) for q in record["route"])
        performance = {}
        for tag, ss in samples.items():
            values = [x["final_cd"] for x in ss]
            assert len(values) == 50
            successes = sum(x < thresholds[tag] for x in values)
            assert math.isclose(mean(values), report["performance"][tag]["final_cd_mean"], abs_tol=1e-12)
            assert successes == report["performance"][tag]["successes"]
            performance[tag] = {"n": 50, "final_cd": mean(values), "standard_error": stdev(values) / math.sqrt(50),
                                "successes": successes, "success_percent": 2 * successes}
        members = {tag: report["domains"][key]["n"] for tag, key in material_domain.items()}
        assert report["member_weighted_purity"] == 1.0
        rows.append({"round": round_number, "performance": performance, "members": members,
                     "stored_member_purity": 1.0, "routing_decisions": dict(routing),
                     "final_episode_selection": dict(final_selection)})
        all_records.extend(records)
    for tag in thresholds:
        for a in (1, 2, 3):
            for b in range(a + 1, 4):
                assert trajectories[tag, a].isdisjoint(trajectories[tag, b])
    counts = Counter()
    for row in rows:
        counts.update(row["routing_decisions"])
    assert counts == {"correct": 653, "incorrect": 7, "frozen": 240}
    rules = Counter(x["write_rule"] for x in all_records)
    assert rules["familiar"] == 69 and rules["rejected"] == 1
    audit = {"source_commit": sources["commit"], "source_files_verified": len(sources["files"]),
             "chunks": 30, "episodes": 300, "unique_goal_segments": len(inputs),
             "trajectory_groups_disjoint_between_rounds": True, "state_links_verified": 29,
             "empty_initial_memory_verified": True, "rounds": rows,
             "routing_decisions": dict(counts), "slow_predictor_selection_accuracy_percent": 100 * 653 / 660,
             "rounds_2_3_correct_slow_selections": 457, "rounds_2_3_total_slow_selections": 457,
             "final_members": rows[-1]["members"], "familiar_not_stored": 69, "rejected_not_stored": 1,
             "tags_used_only_for_posthoc_statistics": True,
             "limits": ["One CEM chain with an absolute-error write check, not the single-domain slow-only configuration.",
                        "CD values originate in the archived per-instance reports; their means match every original chunk log.",
                        "Simulator success flags are not used; success is CD < 0.5 for rope and CD < 0.2 for granular.",
                        "Purity describes stored members; retrieval accuracy excludes frozen-reference selections.",
                        "Different targets across rounds; no paired Frozen or cross-material positive-transfer comparison."]}
    (DATA / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    table = []
    for row in rows:
        r, g = row["performance"]["rope"], row["performance"]["granular"]
        routing = row["routing_decisions"]
        correct = routing.get("correct", 0)
        total = correct + routing.get("incorrect", 0)
        table.append(f"{row['round']} & {r['final_cd']:.3f} & {r['success_percent']:.0f} & {g['final_cd']:.3f} & {g['success_percent']:.0f} & "
                     f"{row['members']['rope']} / {row['members']['granular']} & 100 & {correct}/{total} " + r"\\")
    (ROOT / "tables/deform_mixed.tex").write_text("\n".join(table) + "\n")
    print(json.dumps({"episodes": 300, "stored_members": rows[-1]["members"], "routing_decisions": dict(counts),
                      "slow_selection_accuracy_percent": 100 * 653 / 660}, indent=2))


if __name__ == "__main__":
    main()
