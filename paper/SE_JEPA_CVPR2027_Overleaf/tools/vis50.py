"""Read the pinned external-baseline episode counts used in the main table."""
import csv
import hashlib
import json


def external_results(directory):
    manifest = json.loads((directory / 'vis50_sources.json').read_text())
    path = directory / 'vis50_episodes.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['csv_sha256']
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == sum(f['episodes'] for f in manifest['files'])
    grouped = {}
    for f in manifest['files']:
        subset = [r for r in rows if (r['method'], r['domain']) == (f['method'], f['domain'])]
        assert sorted(int(r['sample_index']) for r in subset) == list(range(50))
        successes = sum(int(r['success']) for r in subset)
        steps = sum(int(r['mpc_steps_with_failure_cap']) for r in subset)
        assert (len(subset), successes, steps) == (f['episodes'], f['successes'], f['steps'])
        grouped[f['method'], f['domain']] = (len(subset), successes, steps)
    methods = {}
    colors = ['red_agent', 'red_object', 'red_goal']
    for method in ['DINO-WM', 'DreamerV3', 'IRIS']:
        shifts = ['blur', 'snp', 'dark', *colors]
        n, k, steps = (sum(grouped[method, d][i] for d in shifts) for i in range(3))
        nc, kc = (sum(grouped[method, d][i] for d in colors) for i in range(2))
        methods[method] = dict(shift_successes=k, shift_episodes=n, color_successes=kc,
                               color_episodes=nc, shift_rate=100*k/n, color_rate=100*kc/nc,
                               steps=steps/n, steps_total=steps)
    return methods


if __name__ == '__main__':
    # Supply a checkout of the baselines commit or its extracted research_snapshots tree.
    import argparse
    from pathlib import Path
    import numpy as np

    cli = argparse.ArgumentParser(description='Verify archived episode rows against the pinned original NPZ files.')
    cli.add_argument('checkout', type=Path, help='root containing research_snapshots/')
    args = cli.parse_args()
    directory = Path(__file__).resolve().parents[1] / 'data/vis50'
    metadata = json.loads((directory/'vis50_sources.json').read_text())
    with (directory/'vis50_episodes.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for source in metadata['files']:
        raw = args.checkout/source['path']
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == source['sha256']
        selected = sorted((r for r in rows if (r['method'], r['domain']) == (source['method'], source['domain'])),
                          key=lambda r: int(r['sample_index']))
        with np.load(raw, allow_pickle=False) as archive:
            success = archive['successes'].astype(bool)
            steps = np.where(success, archive['action_len'], 30)
            assert np.array_equal(success.astype(int), [int(r['success']) for r in selected])
            assert np.array_equal(steps, [int(r['mpc_steps_with_failure_cap']) for r in selected])
    external_results(directory)
    print(f"Matched {len(rows)} episode rows to the {len(metadata['files'])} original NPZ files.")
