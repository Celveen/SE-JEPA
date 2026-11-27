"""Build the deployment streams of the PushObj (shape shifts) and PushT (visual shifts) experiments.

A stream is a directory data/streams/<name>/ with one chunk file per block (plan_targets.pkl format, planned
in stored order) and a manifest.json that lists the blocks in stream order. Each block carries its planning
seed and, for visual streams, the Hydra overrides that apply its visual condition.

Shape streams (PushObj). Round r uses seed seeds[r]; for every shape (in the given order) the chunk holds the
n targets that random.Random(seed).sample draws from data/pushobj_eval/val_<shape>/plan_targets.pkl.

    python scripts/build_streams.py shape --name shape_s100 --seeds 100 200 300 1000 1100
    python scripts/build_streams.py shape --name shape_s400 --seeds 400 500 600 1200 1300
    python scripts/build_streams.py shape --name shape_s700 --seeds 700 800 900 1400 1500
    # held-out test: targets that no listed stream uses
    python scripts/build_streams.py shape --name shape_test --seeds 2000 2100 2200 \
        --exclude shape_s100 shape_s400 shape_s700

Visual streams (PushT). Targets come from data/pushobj_eval/val_T/plan_targets.pkl. Round r draws n targets
with random.Random(seeds[r]); the seven visual conditions of a round share these targets and run in the fixed
order default, blur, snp1, dark, red_agent, red_object, red_goal.

    python scripts/build_streams.py visual --name visual_s100 --seeds 100 200 300 1000 1100
    python scripts/build_streams.py visual --name visual_s400 --seeds 400 500 600 1200 1300
    python scripts/build_streams.py visual --name visual_s700 --seeds 700 800 900 1400 1500
    # held-out test: one round per seed, source trajectories (ep_idx) disjoint from the listed streams and
    # from each other
    python scripts/build_streams.py visual --name visual_test --seeds 2000 2100 2200 --heldout \
        --exclude visual_s100 visual_s400 visual_s700
"""
import argparse
import json
import pickle
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ["I", "L", "square", "Z", "small_tee", "+", "T"]
VISUAL = {  # condition -> Hydra overrides, in stream order
    "default": [],
    "blur": ["ood_corruption=blur", "ood_level=2.0"],
    "snp1": ["ood_corruption=snp1", "ood_level=0.01"],
    "dark": ["ood_corruption=dark", "ood_level=0.9"],
    "red_agent": ["++env_kwargs_override.agent_color=Red"],
    "red_object": ["++env_kwargs_override.color=Red"],
    "red_goal": ["++env_kwargs_override.goal_color=Red"],
}


def load_segments(data, shape):
    return pickle.load(open(ROOT / data / f"val_{shape}" / "plan_targets.pkl", "rb"))["segments"]


def excluded(out_root, names):
    """Segment indices (per shape) and source trajectories (ep_idx) used by the listed streams."""
    used, sources = {}, set()
    for name in names:
        for b in json.loads((out_root / name / "manifest.json").read_text())["blocks"]:
            used.setdefault(b["shape"], set()).update(b["indices"])
            sources.update(b["ep_idx"])
    return used, sources


def write_chunk(out, blocks, segs, idx, shape, tag, seed, overrides):
    chunk = f"chunk_{len(blocks):02d}_{tag}"
    with open(out / f"{chunk}.pkl", "wb") as f:
        pickle.dump({"segments": [segs[i] for i in idx], "traj_len": 25, "shape": shape,
                     "n_samples": len(idx), "seed": seed, "indices": idx}, f)
    blocks.append({"chunk": chunk, "shape": shape, "tag": tag, "n": len(idx), "seed": seed, "indices": idx,
                   "ep_idx": [segs[i]["ep_idx"] for i in idx], "overrides": overrides})
    print(f"{chunk}: {len(idx)} targets, seed {seed}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["shape", "visual"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="one seed per round")
    ap.add_argument("--n", type=int, default=50, help="targets per chunk")
    ap.add_argument("--shapes", nargs="+", default=SHAPES, help="shape streams: shapes per round, in order")
    ap.add_argument("--exclude", nargs="*", default=[], help="streams whose targets this stream leaves out")
    ap.add_argument("--heldout", action="store_true",
                    help="visual streams: also exclude the source trajectories of the listed streams, and keep "
                         "the source trajectories of different seeds disjoint")
    ap.add_argument("--data", default="data/pushobj_eval")
    ap.add_argument("--out", default="data/streams")
    a = ap.parse_args()

    out_root = ROOT / a.out
    used, used_sources = excluded(out_root, a.exclude)
    out = out_root / a.name
    out.mkdir(parents=True, exist_ok=True)
    blocks = []
    if a.kind == "shape":
        for seed in a.seeds:
            for shape in a.shapes:
                segs = load_segments(a.data, shape)
                pool = [i for i in range(len(segs)) if i not in used.get(shape, ())]
                idx = [pool[j] for j in random.Random(seed).sample(range(len(pool)), min(a.n, len(pool)))]
                write_chunk(out, blocks, segs, idx, shape, shape.replace("+", "plus"), seed, [])
    else:
        segs = load_segments(a.data, "T")
        taken = set(used_sources) if a.heldout else set()
        for seed in a.seeds:
            if a.heldout:  # one target per source trajectory, none shared with the excluded streams or other seeds
                order = list(range(len(segs)))
                random.Random(seed).shuffle(order)
                idx, seen = [], set()
                for i in order:
                    ep = segs[i]["ep_idx"]
                    if ep in taken or ep in seen:
                        continue
                    idx.append(i)
                    seen.add(ep)
                    if len(idx) == a.n:
                        break
                taken |= seen
            else:
                pool = [i for i in range(len(segs)) if i not in used.get("T", ())]
                idx = [pool[j] for j in random.Random(seed).sample(range(len(pool)), min(a.n, len(pool)))]
            for cond, overrides in VISUAL.items():
                write_chunk(out, blocks, segs, idx, "T", cond, seed, overrides)
    (out / "manifest.json").write_text(json.dumps({"name": a.name, "kind": a.kind, "seeds": a.seeds,
                                                   "blocks": blocks}, indent=1))


if __name__ == "__main__":
    main()
