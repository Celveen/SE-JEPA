"""Extract the frames of a PushT/PushObj dataset to PNG for DDP-WM training (its frame_format=png).

DDP-WM's loader reads obses_png/episode_XXX/NNNN.png; decoding each mp4 once and saving lossless PNGs keeps the
pixels the video decoder produces, so the model sees the same images as from the videos, only faster.
    python extract_png.py <dataset dir> [workers]
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from decord import VideoReader
from PIL import Image


def one(args):
    mp4, out = args
    out.mkdir(parents=True, exist_ok=True)
    r = VideoReader(str(mp4), num_threads=1)
    frames = r.get_batch(list(range(len(r)))).asnumpy()
    for t, f in enumerate(frames):
        p = out / f"{t:04d}.png"
        if not p.exists():
            Image.fromarray(np.ascontiguousarray(f)).save(p, compress_level=1)
    return len(frames)


def main():
    root = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    for split in ("train", "val"):
        vids = sorted((root / split / "obses").glob("episode_*.mp4"))
        jobs = [(v, root / split / "obses_png" / v.stem) for v in vids]
        with Pool(workers) as pool:
            n = sum(pool.imap_unordered(one, jobs, chunksize=16))
        print(f"{split}: {len(vids)} episodes, {n} frames", flush=True)


if __name__ == "__main__":
    main()
