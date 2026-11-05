import argparse
from pathlib import Path

from common import rpc_serve
from dreamer_model import DreamerScorer


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--socket", type=Path, required=True)
arguments = parser.parse_args()
model = DreamerScorer(arguments.checkpoint)
rpc_serve(arguments.socket, model)
