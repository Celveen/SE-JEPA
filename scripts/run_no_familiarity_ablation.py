"""Run one 50-episode both-off batch, with explicit input and output paths.

Learning batches must be chained using --state-in. Held-out batches all load
the same final learning state and use --read-only. No jobs run on import.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--planner', choices=('gd', 'cem'), required=True)
    result.add_argument('--checkpoint', type=Path, required=True,
                        help='Directory containing hydra.yaml and checkpoints/model_latest.pth')
    result.add_argument('--segments', type=Path, required=True,
                        help='The original ordered 50-episode target PKL for this batch')
    result.add_argument('--output', type=Path, required=True,
                        help='New or empty batch output directory')
    result.add_argument('--seed', type=int, required=True)
    result.add_argument('--tag', required=True,
                        help='Domain label for diagnostics, e.g. I, small_tee or plus')
    result.add_argument('--state-in', type=Path)
    result.add_argument('--read-only', action='store_true',
                        help='Test from state-in without updating persistent slow state')
    result.add_argument('--dry-run', action='store_true',
                        help='Validate paths and print the command without running the model')
    return result


def quoted(value):
    # Hydra strings need their own quotes even though subprocess uses no shell.
    return json.dumps(str(value), ensure_ascii=False)


def build_command(args):
    checkpoint = args.checkpoint.expanduser().resolve()
    segments = args.segments.expanduser().resolve()
    output = args.output.expanduser().resolve()
    state_in = args.state_in.expanduser().resolve() if args.state_in else None
    if args.read_only and state_in is None:
        raise ValueError('--read-only requires --state-in pointing to the final learning state')
    for path in (checkpoint / 'hydra.yaml', checkpoint / 'checkpoints/model_latest.pth', segments):
        if not path.is_file():
            raise ValueError('Missing required input: {}'.format(path))
    if state_in is not None and not state_in.is_file():
        raise ValueError('Missing state-in: {}'.format(state_in))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be a new or empty directory: {}'.format(output))
    return [
        sys.executable, str(ROOT / 'plan.py'),
        '--config-name', 'ablation_no_familiarity_{}_pushobj'.format(args.planner),
        'ckpt_base_path=' + quoted(checkpoint.as_posix()),
        'model_epoch=latest', 'eval_data_path=' + quoted(segments.as_posix()),
        'hydra.run.dir=' + quoted(output.as_posix()),
        'seed={}'.format(args.seed), 'n_evals=50',
        'segments_order=file', 'goal_source=segments',
        'planner.continual.tag=' + quoted(args.tag),
        'planner.continual.state_in=' + (quoted(state_in.as_posix()) if state_in else 'null'),
        'planner.continual.state_out=continual_state.pt',
        'planner.continual.read_only=' + str(args.read_only).lower(),
    ]


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        command = build_command(args)
    except ValueError as error:
        cli.error(str(error))
    print(' '.join(shlex.quote(item) for item in command), flush=True)
    if args.dry_run:
        return 0
    # plan.py/Hydra owns its output directory. No shell, downloads or job queues.
    return subprocess.run(command, cwd=str(ROOT)).returncode


if __name__ == '__main__':
    sys.exit(main())
