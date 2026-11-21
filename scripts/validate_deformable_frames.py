"""Independently check recorded PNGs and particle-space Chamfer distances."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path, help='Packaged media directory')
parser.add_argument('--min-groups', type=int, default=8)
args = parser.parse_args()
root = args.root.resolve()
manifest = json.loads((root / 'manifest.json').read_text())
assert len(manifest) >= args.min_groups
checks = []
for group in manifest:
    folder = root / group['group']
    assert hashlib.sha256((folder / 'target.pkl').read_bytes()).hexdigest() == group['target_sha256']
    with (folder / 'target.pkl').open('rb') as stream:
        target = pickle.load(stream)
    goal = np.asarray(target['state_g'])[0].reshape(-1, 4)[..., :3]
    paired_start = []
    for arm in (group['before_arm'], group['after_arm']):
        path = folder / arm
        receipt = json.loads((path / 'complete.json').read_text())
        assert receipt['env_seed'] == group['source']['env_seed']
        assert receipt['plan_seed'] == group['source']['plan_seed']
        if arm in ('round1', 'round3'):
            applied = json.loads((path / 'learner_applied.json').read_text())
            assert applied['episode'] == (50 if arm == 'round1' else 150)
            assert applied['domain'] == group['source']['object']
        records = json.loads((path / 'frames.json').read_text())
        assert len(records) == receipt['motion_frames'] == len(list((path / 'motion_frames').glob('*.png')))
        assert len(list((path / 'keyframes').glob('*.png'))) == 16
        for record in records:
            rgb = np.asarray(Image.open(path / 'motion_frames' / record['file']).convert('RGB'))
            assert rgb.shape == (224, 224, 3)
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == record['rgb_sha256']
        for push in range(1, 16):
            last = [record for record in records if record['push'] == push][-1]
            key = np.asarray(Image.open(path / 'keyframes' / f'push_{push:02d}.png').convert('RGB'))
            assert hashlib.sha256(key.tobytes()).hexdigest() == last['rgb_sha256']
        states = np.load(path / 'particle_states.npz')['states']
        assert len(states) == 16 and np.isfinite(states).all()
        assert np.array_equal(states[0], np.asarray(target['state_0'])[0].reshape(-1, 4))
        paired_start.append(np.asarray(Image.open(path / 'keyframes/push_00.png')))
        assert np.array_equal(np.asarray(Image.open(path / 'goal.png')), np.asarray(target['obs_g']['visual'])[0, 0])
        errors = []
        for push in (5, 10, 15):
            current = states[push, ..., :3].reshape(-1, 3)
            cd = cKDTree(goal).query(current)[0].mean() + cKDTree(current).query(goal)[0].mean()
            error = abs(cd - receipt[f'CD{push}'])
            assert error < 1e-5, (group['group'], arm, push, cd, receipt[f'CD{push}'])
            errors.append(float(error))
        checks.append({'group': group['group'], 'arm': arm, 'motion_frames': len(records), 'max_CD_error': max(errors)})
    start_mae = float(np.abs(paired_start[0].astype(float)-paired_start[1]).mean())
    assert start_mae < 2, (group['group'], start_mae)
    checks[-1]['paired_start_rgb_mae'] = start_mae
    if group['comparison'] == 'continual':
        with np.load(folder / 'round1_online/source_episode.npz') as original:
            assert np.array_equal(original['state_0'], np.asarray(target['state_0'])[0])
            assert np.array_equal(original['state_g'], np.asarray(target['state_g'])[0])
        online_start = np.asarray(Image.open(folder / 'round1_online/keyframes/push_00.png'))
        online_mae = float(np.abs(online_start.astype(float)-paired_start[0]).mean())
        assert online_mae < 2, (group['group'], online_mae)
        checks[-1]['online_start_rgb_mae'] = online_mae
(root / 'validation.json').write_text(json.dumps({'passed': True, 'checks': checks}, indent=2))
print(f'Validated {len(manifest)} groups and {sum(c["motion_frames"] for c in checks)} native motion frames.')
