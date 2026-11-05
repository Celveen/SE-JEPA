import argparse
from pathlib import Path

from common import rpc_serve
from iris_model import IrisScorer


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--socket", type=Path, required=True)
arguments = parser.parse_args()
model = IrisScorer(arguments.checkpoint_dir)
rpc_serve(arguments.socket, model)
