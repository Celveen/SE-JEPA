"""Verify the archived seven-shape CEM evaluations before rendering table cells."""
from pathlib import Path
import csv, hashlib, json
import numpy as np


def verified_pushobj(root):
    root = Path(root)
    source = json.loads((root / 'git_source.json').read_text())
    declared = json.loads((root / 'results.json').read_text())['methods']
    expected = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.removeprefix('./')] = digest
    hashes_verified = 0
    for name in source['files']:
        if name == 'SHA256SUMS':
            continue
        assert name in expected, name
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected[name], name
        hashes_verified += 1
    csv_rows = list(csv.DictReader((root / 'episodes.csv').open()))
    states, results = {}, {}
    for method, label in [('dreamerv3', 'DreamerV3'), ('iris', 'IRIS')]:
        all_success, all_steps, unseen = [], [], []
        for shape in ['I', 'L', 'square', 'Z', 'small_tee', 'plus', 'T']:
            folder = root / 'completed_runs' / method / shape
            with np.load(folder / 'final_eval_results.npz', allow_pickle=False) as z:
                actual = z['successes']
                distance = np.linalg.norm(z['state_g'][:, :4] - z['final_states'][:, :4], axis=1)
                angle = np.abs(z['state_g'][:, 4] - z['final_states'][:, 4]) % (2 * np.pi)
                angle = np.minimum(angle, 2 * np.pi - angle)
                np.testing.assert_array_equal(actual, (distance < 20) & (angle < np.pi / 9))
                steps = z['executed_mpc_steps']
                assert len(actual) == 50 and np.all((steps >= 1) & (steps <= 30))
                assert np.all(steps[~actual] == 30)
                np.testing.assert_array_equal(z['physical_action_count'], 5 * steps)
                np.testing.assert_array_equal(z['action_len'][actual], steps[actual])
                key = shape
                if key in states:
                    for name in ('state_0', 'state_g'):
                        np.testing.assert_array_equal(z[name], states[key][name])
                else:
                    states[key] = {name: z[name].copy() for name in ('state_0', 'state_g')}
                rows = [x for x in csv_rows if x['method'] == method and x['shape'] == ('+' if shape == 'plus' else shape)]
                assert len(rows) == 50
                rows.sort(key=lambda x: int(x['sample_index']))
                np.testing.assert_array_equal(actual, [int(x['success']) for x in rows])
                np.testing.assert_array_equal(steps, [int(x['mpc_steps']) for x in rows])
                all_success.extend(actual.tolist()); all_steps.extend(steps.tolist())
                if shape in ('I', 'square', 'small_tee'): unseen.extend(actual.tolist())
            frozen = json.loads((folder / 'frozen_weights_check.json').read_text())
            assert frozen['unchanged'] and frozen['before_sha256'] == frozen['after_sha256']
            protocol = json.loads((folder / 'protocol.json').read_text())
            assert protocol['frozen_model'] and protocol['training_shapes'] == ['T']
        assert len(all_success) == 350 and len(unseen) == 150
        result = {'episodes': 350, 'successes': sum(all_success), 'all_percent': np.mean(all_success) * 100,
                  'unseen_episodes': 150, 'unseen_successes': sum(unseen), 'unseen_percent': np.mean(unseen) * 100,
                  'mean_steps': np.mean(all_steps)}
        assert result['successes'] == declared[method]['successes']
        assert abs(result['mean_steps'] - declared[method]['mean_mpc_steps']) < 1e-10
        assert abs(result['unseen_percent'] / 100 - declared[method]['I_square_small_tee_success_rate']) < 1e-10
        results[label] = result
    audit = {'source_commit': source['commit'], 'source_branch': source['branch'],
             'verified_result_files': 14, 'verified_episodes': 700, 'source_file_hashes_verified': hashes_verified,
             'success_recomputed_from_final_states': True, 'matched_initial_goal_states_between_methods': True,
             'failure_budget': 30, 'actions_per_step': 5, 'all_checkpoints_frozen': True,
             'training_shapes': ['T'], 'main_table_unseen_group': ['I', 'square', 'small_tee'], 'results': results}
    (root / 'local_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return results


if __name__ == '__main__':
    print(json.dumps(verified_pushobj(Path(__file__).resolve().parents[1] / 'data/dreamer_iris_pushobj_20261110'), indent=2))
