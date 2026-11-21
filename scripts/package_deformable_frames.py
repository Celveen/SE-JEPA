"""Select eight qualitative pairs and package their native simulator recordings."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--objects", nargs="+", choices=("rope", "granular"), default=["rope", "granular"])
args = parser.parse_args()
source = args.source.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[1]
hashes = {}
def digest(path):
    path = Path(path)
    if str(path) not in hashes:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                hasher.update(chunk)
        hashes[str(path)] = hasher.hexdigest()
    return hashes[str(path)]
def read_image(path):
    return np.asarray(Image.open(path).convert("RGB"))

candidates = []
screening = []
for folder in sorted((source / "candidates").iterdir()):
    data = json.loads((folder / "source.json").read_text())
    arms = {}
    for arm in ("round1", "round3", "frozen"):
        receipt = folder / arm / "complete.json"
        if not receipt.exists():
            continue
        arms[arm] = json.loads(receipt.read_text())
        screening.append({
            "case": data["case"], "object": data["object"],
            "arm": arm, "source_round": data["online_row"]["round"],
            "CD5": arms[arm]["CD5"], "CD10": arms[arm]["CD10"], "CD15": arms[arm]["CD15"],
            "trajectory": data["trajectory"], "offset": data["offset"],
            "plan_seed": data["plan_seed"], "environment_seed": data["env_seed"],
        })
    if len(arms) == 3:
        candidates.append({"source": data, "arms": arms, "folder": folder})
with (output / "screening.csv").open("w") as stream:
    writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=list(screening[0]))
    writer.writeheader()
    writer.writerows(screening)

chosen = []
for obj in args.objects:
    pool = [c for c in candidates if c["source"]["object"] == obj]
    used = set()
    for kind, before in (("continual", "round1"), ("baseline", "frozen")):
        eligible = [c for c in pool if c["source"]["case"] not in used
                    and c["arms"]["round3"]["CD15"] < 0.7 * c["arms"][before]["CD15"]]
        if kind == "continual":
            # The main comparison is the archived first encounter against the
            # round-3 checkpoint on the same goal. Also require improvement
            # over the fixed round-1 checkpoint in the matched-seed evaluation.
            eligible = [c for c in pool if c["source"]["case"] not in used
                        and c["source"]["candidate_reason"] == "round1_high_CD"
                        and c["arms"]["round3"]["CD15"] < 0.7 * float(c["source"]["online_row"]["CD15"])
                        and c["arms"]["round3"]["CD15"] < c["arms"]["round1"]["CD15"]]
        eligible.sort(key=lambda c: (float(c["source"]["online_row"]["CD15"]) if kind == "continual" else c["arms"][before]["CD15"]) - c["arms"]["round3"]["CD15"], reverse=True)
        if len(eligible) < 2:
            raise RuntimeError(f"Need two clear {obj} {kind} pairs; found {len(eligible)}")
        for index, candidate in enumerate(eligible[:2], start=1):
            used.add(candidate["source"]["case"])
            chosen.append((f"{obj}_{kind}_{index:02d}", kind, before, candidate))

published = []
labels = {"round1": "SE-JEPA after round 1", "round3": "SE-JEPA after round 3", "frozen": "Frozen DINO-WM"}
for name, kind, before, candidate in chosen:
    folder = output / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    src = candidate["source"]
    shutil.copy2(candidate["folder"] / "target.pkl", folder / "target.pkl")
    cfgs = {}
    for arm in (before, "round3"):
        origin = candidate["folder"] / arm
        destination = folder / arm
        shutil.copytree(origin / "motion_frames", destination / "motion_frames", dirs_exist_ok=True)
        shutil.copytree(origin / "keyframes", destination / "keyframes", dirs_exist_ok=True)
        for filename in ("goal.png", "frames.json", "complete.json", "particle_states.npz",
                         "final_eval_results.npz", "logs.json", "preprocessing.json"):
            shutil.copy2(origin / filename, destination / filename)
        if (origin / "learner_applied.json").exists():
            shutil.copy2(origin / "learner_applied.json", destination / "learner_applied.json")
        records = json.loads((destination / "frames.json").read_text())
        assert len(list((destination / "motion_frames").glob("*.png"))) == len(records)
        for record in records:
            rgb = read_image(destination / "motion_frames" / record["file"])
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == record["rgb_sha256"]
        cfg = json.loads((origin / "config.json").read_text())
        cfgs[arm] = cfg
        runtime = {
            "action_file": str(repo / "eval_outputs/single_domain_new50x3/inputs" / f"{src['object']}_actions.pth"),
            "dino_hub": "/data/zjc/JEPA-Stall/eval_outputs/rope_local/torch/hub/facebookresearch_dinov2_main",
            "dino_weights": "/data/zjc/JEPA-Stall/eval_outputs/rope_local/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
        }
        cfg["runtime"] = runtime
        cfg["media"]["runtime_launcher"] = str(repo / "scripts/run_deformable_file_targets.py")
        cfg["goal_file_path"] = str(folder / "target.pkl")
        cfg["saved_folder"] = str(repo / "eval_outputs/community_frames/reproduced" / name / arm)
        (destination / "reproduce.json").write_text(json.dumps(cfg, indent=2))
    assert np.array_equal(read_image(folder / before / "goal.png"), read_image(folder / "round3" / "goal.png"))
    metadata = {
        "group": name, "comparison": kind, "source": src,
        "before_arm": before, "after_arm": "round3",
        "before_CD": candidate["arms"][before]["CD15"],
        "after_CD": candidate["arms"]["round3"]["CD15"],
        "relative_CD_reduction": 1-candidate["arms"]["round3"]["CD15"]/candidate["arms"][before]["CD15"],
        "target_sha256": digest(folder / "target.pkl"),
        "checkpoint_sha256": {},
        "rendering": "Native PyFleX RGB, 224x224; captured from existing store_data calls during planning.",
    }
    for arm, cfg in cfgs.items():
        checkpoint = cfg["planner"]["continual"]["state_in"] or str(Path(cfg["ckpt_base_path"]) / "checkpoints/model_latest.pth")
        metadata["checkpoint_sha256"][arm] = digest(checkpoint)
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.8))
    for row, arm in enumerate((before, "round3")):
        for col, step in enumerate((0, 5, 10, 15)):
            axes[row, col].imshow(read_image(folder / arm / "keyframes" / f"push_{step:02d}.png"), interpolation="nearest")
            axes[row, col].set_title("Start" if step == 0 else f"Push {step}", fontsize=10)
        axes[row, 4].imshow(read_image(folder / arm / "goal.png"), interpolation="nearest")
        axes[row, 4].set_title("Goal", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
        axes[row, 0].set_ylabel(labels[arm], fontsize=10)
        axes[row, 3].set_xlabel(f"Final CD {candidate['arms'][arm]['CD15']:.3f}", fontsize=10)
    fig.suptitle(f"{src['object'].capitalize()} | {metadata['relative_CD_reduction']:.0%} lower final CD", fontsize=14)
    fig.tight_layout()
    fig.savefig(folder / "storyboard.png", dpi=140)
    plt.close(fig)
    if kind == "continual":
        online = folder / "round1_online"
        (online / "keyframes").mkdir(parents=True, exist_ok=True)
        episode = int(src["online_row"]["episode"])
        recorded = repo / "eval_outputs/single_domain_new50x3" / src["object"] / "round_01" / src["source_block"] / "episode_frames.npz"
        with np.load(recorded) as archive:
            rgb = archive[f"ep{episode}_visual"]
            assert rgb.shape == (16, 224, 224, 3)
            for step, frame in enumerate(rgb):
                Image.fromarray(frame.astype(np.uint8)).save(online / "keyframes" / f"push_{step:02d}.png")
        shutil.copy2(folder / "round3" / "goal.png", online / "goal.png")
        with np.load(recorded.parent / "final_eval_results.npz") as archive:
            np.savez_compressed(online / "source_episode.npz", **{key: archive[key][src["sample"]] for key in ("state_0", "state_g", "actions")})
        online_cd = float(src["online_row"]["CD15"])
        online_meta = {
            "episode": episode, "source_chunk": src["source_block"],
            "source_sha256": digest(recorded), "CD15": online_cd,
            "capture": "Original online round-1 simulation observations at push boundaries.",
        }
        (online / "metadata.json").write_text(json.dumps(online_meta, indent=2))
        metadata["round1_online_CD"] = online_cd
        metadata["online_to_round3_relative_CD_reduction"] = 1-metadata["after_CD"]/online_cd
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (folder / "storyboard.png").rename(folder / "checkpoint_comparison.png")
        fig, axes = plt.subplots(2, 5, figsize=(12, 5.8))
        for row, (arm, label, cd) in enumerate([
            ("round1_online", "First encounter in round 1", online_cd),
            ("round3", "SE-JEPA after round 3", metadata["after_CD"]),
        ]):
            for col, step in enumerate((0, 5, 10, 15)):
                axes[row, col].imshow(read_image(folder / arm / "keyframes" / f"push_{step:02d}.png"), interpolation="nearest")
                axes[row, col].set_title("Start" if step == 0 else f"Push {step}", fontsize=10)
            axes[row, 4].imshow(read_image(folder / arm / "goal.png"), interpolation="nearest")
            axes[row, 4].set_title("Goal", fontsize=10)
            axes[row, 0].set_ylabel(label, fontsize=10)
            axes[row, 3].set_xlabel(f"Final CD {cd:.3f}", fontsize=10)
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        fig.suptitle(f"{src['object'].capitalize()} | same start and goal", fontsize=14)
        fig.tight_layout()
        fig.savefig(folder / "storyboard.png", dpi=140)
        plt.close(fig)
    published.append(metadata)

(output / "manifest.json").write_text(json.dumps(published, indent=2))
with (output / "comparisons.csv").open("w") as stream:
    writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=["group", "before_arm", "before_CD", "after_CD", "relative_CD_reduction", "round1_online_CD", "online_to_round3_relative_CD_reduction"])
    writer.writeheader()
    for item in published:
        writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
print(json.dumps([{k: item[k] for k in ("group", "before_CD", "after_CD")} for item in published], indent=2))
