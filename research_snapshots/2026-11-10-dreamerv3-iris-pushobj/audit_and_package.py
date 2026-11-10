"""Audit the completed frozen PushObj evaluations and prepare a focused publication."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PLAN = Path('/data/zjc/adaJEPA/experiment_plans/dreamerv3_iris_pushobj_H7t3_proposal_20261110')
SNAPSHOT = HERE / 'snapshot'
RELATIVE_ROOT = 'research_snapshots/2026-11-10-dreamerv3-iris-pushobj'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    manifest = json.loads((PLAN / 'manifest.json').read_text())
    output = Path(manifest['output_root'])
    state = json.loads((output / 'pipeline_status.json').read_text())
    assert state['status'] == 'completed' and state['episodes_saved'] == state['planned_episodes'] == 700
    assert len(state['jobs']) == 8 and all(j['exit_code'] == 0 and j['status'] == 'completed' for j in state['jobs'].values())
    source_code = {}
    for relative, expected in manifest['prepared_code_sha256'].items():
        assert sha(PLAN / relative) == expected, relative
        source_code[relative] = expected
    for path, expected in manifest['runtime_dependency_sha256'].items():
        assert sha(path) == expected, path
    blocks, methods, fingerprints, table_rows, episode_rows = [], {}, {}, [], []
    for method, model in manifest['models'].items():
        for name, expected in model['checkpoint_sha256'].items():
            assert sha(Path(model['checkpoint']) / name) == expected, name
        assert json.loads((Path(model['checkpoint']) / 'training_complete.json').read_text())['epoch'] == 3
        original_summary = json.loads((output / method / 'summary.json').read_text())
        by_shape, method_fingerprints, all_steps = {}, set(), []
        for block in manifest['blocks']:
            folder = output / method / block['tag']
            protocol = json.loads((folder / 'protocol.json').read_text())
            assert protocol['method'] == method and protocol['shape'] == block['shape']
            assert protocol['planning_seed'] == 2000 and protocol['episodes'] == 50
            assert protocol['CEM'] == dict(samples=200, topk=30, iterations=10, model_horizon=5, max_replans=30)
            assert protocol['frameskip'] == 5 and protocol['frozen_model'] and not protocol['diagnostic_only']
            assert protocol['noise'] == 'none for clean PushObj shapes'
            hashes = model['checkpoint_sha256']
            expected_checksum = hashes['model_latest.pkl'] if method == 'dreamerv3' else hashlib.sha256(
                (hashes['world_model.pt'] + hashes['tokenizer_model.pt']).encode()).hexdigest()
            assert protocol['checkpoint_sha256'] == expected_checksum
            assert protocol['target_sha256'] == sha(block['target']) == block['target_sha256']
            config = yaml.safe_load((folder / '.hydra/config.yaml').read_text())
            assert config['model_epoch'] == 3 and config['seed'] == 2000 and config['n_evals'] == 50
            frozen = json.loads((folder / 'frozen_weights_check.json').read_text())
            assert frozen['unchanged'] and frozen['before_sha256'] == frozen['after_sha256']
            method_fingerprints.add(frozen['before_sha256'])
            logs = defaultdict(list)
            for line in (folder / 'per_iter_logs.json').read_text().splitlines():
                row = json.loads(line)
                assert 0 <= row['sample_idx'] < 50 and 1 <= row['step'] <= 30
                assert math.isfinite(row['goal_pixel_proprio_cost']) and math.isfinite(row['state_dist'])
                logs[row['sample_idx']].append(row)
            with np.load(folder / 'final_eval_results.npz', allow_pickle=False) as data, np.load(block['target'], allow_pickle=False) as target:
                assert data['successes'].shape == (50,) and data['actions'].shape == (50, 30, 10)
                assert np.isfinite(data['actions']).all() and np.isfinite(data['final_states']).all()
                for key in ['state_0', 'state_g']:
                    np.testing.assert_array_equal(data[key], target[key])
                np.testing.assert_array_equal(target['seeds'], 1 + 2000 * np.arange(50))
                steps = data['executed_mpc_steps']
                assert np.all((steps >= 1) & (steps <= 30))
                np.testing.assert_array_equal(data['physical_action_count'], 5 * steps)
                np.testing.assert_array_equal(steps, np.where(data['successes'], data['action_len'], 30))
                pos = np.linalg.norm(data['state_g'][:, :4] - data['final_states'][:, :4], axis=1)
                angle = np.abs((data['state_g'][:, 4] - data['final_states'][:, 4] + np.pi) % (2 * np.pi) - np.pi)
                np.testing.assert_array_equal((pos < 20) & (angle < np.pi / 9), data['successes'])
                episodes = sorted((folder / 'episodes').glob('episode_*.npz'))
                assert len(episodes) == 50
                for sample in range(50):
                    rows = logs[sample]
                    assert [r['step'] for r in rows] == list(range(1, int(steps[sample]) + 1))
                    assert not any(r['success'] for r in rows[:-1]) and bool(rows[-1]['success']) == bool(data['successes'][sample])
                    with np.load(folder / f'episodes/episode_{sample:02d}.npz') as ep:
                        for ep_name, final_name in [('actions', 'actions'), ('final_state', 'final_states'),
                                                   ('success', 'successes'), ('action_len', 'action_len'),
                                                   ('executed_mpc_steps', 'executed_mpc_steps'), ('physical_action_count', 'physical_action_count')]:
                            np.testing.assert_array_equal(ep[ep_name], data[final_name][sample])
                    episode_rows.append(dict(method=method, shape=block['shape'], sample_index=sample,
                        environment_seed=int(target['seeds'][sample]), planning_seed=2000 + sample,
                        success=int(data['successes'][sample]), mpc_steps=int(steps[sample]),
                        physical_actions=int(5 * steps[sample]), joint_position_error=float(pos[sample]), angle_error_radians=float(angle[sample])))
                result = dict(n=50, successes=int(data['successes'].sum()), success_rate=float(data['successes'].mean()),
                              mean_mpc_steps=float(steps.mean()), mean_physical_actions=float((5 * steps).mean()))
                assert result == original_summary['by_shape'][block['shape']]
                by_shape[block['shape']] = result
                all_steps.extend(steps.tolist())
            blocks.append(dict(method=method, shape=block['shape'], episodes=50,
                               verified_complete=True, independently_recomputed_success_matches=True,
                               iteration_logs_match_action_counts=True, frozen=True))
            table_rows.append(dict(method=method, shape=block['shape'], **result))
        assert len(method_fingerprints) == 1, 'Parallel workers loaded different model tensors'
        fingerprints[method] = next(iter(method_fingerprints))
        summary = dict(episodes=350, successes=sum(x['successes'] for x in by_shape.values()),
            all_success_rate=sum(x['successes'] for x in by_shape.values()) / 350,
            T_seen_success_rate=by_shape['T']['success_rate'],
            six_unseen_success_rate=sum(x['successes'] for k, x in by_shape.items() if k != 'T') / 300,
            I_square_small_tee_success_rate=sum(by_shape[k]['successes'] for k in ['I', 'square', 'small_tee']) / 150,
            mean_mpc_steps=float(np.mean(all_steps)), mean_physical_actions=float(5 * np.mean(all_steps)), by_shape=by_shape)
        for key in ['all_success_rate', 'T_seen_success_rate', 'six_unseen_success_rate', 'I_square_small_tee_success_rate']:
            assert math.isclose(summary[key], original_summary[key], abs_tol=1e-12)
        methods[method] = summary
    assert len(blocks) == 14 and len(episode_rows) == 700
    elapsed = (datetime.fromisoformat(state['ended_at']) - datetime.fromisoformat(state['started_at'])).total_seconds()
    audit = dict(status='passed', audited_at_UTC=datetime.now(timezone.utc).isoformat(), episodes=700, blocks=blocks,
        methods=methods, frozen_tensor_fingerprints=fingerprints, elapsed_wall_seconds=elapsed,
        checks=['700 saved episode files match final result arrays', '14 x 50 complete targets and fixed seed mapping',
                'Success independently recomputed from final states', 'MPC iteration logs and physical action counts agree',
                'Checkpoint, target, prepared code and runtime dependency hashes unchanged',
                'Frozen tensor fingerprints identical before/after and across parallel workers', 'All eight jobs exited zero'])
    assert not SNAPSHOT.exists(), 'Existing archive; inspect before rewriting'
    SNAPSHOT.mkdir()
    copied = []
    def copy_file(source, destination):
        source, destination = Path(source), Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert sha(source) == sha(destination)
        copied.append(dict(source=str(source), stored=str(destination.relative_to(SNAPSHOT)), sha256=sha(destination), bytes=destination.stat().st_size))
    def copy_tree(source, destination):
        for path in sorted(Path(source).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in ['.pyc', '.lock']:
                continue
            copy_file(path, Path(destination) / path.relative_to(source))
    copy_tree(output, SNAPSHOT / 'completed_runs')
    for name in ['README.md', 'manifest.json', 'run.py', 'run_all.py', 'validate_preflight.py',
                 'preflight_validation.json', 'launch_record.json', 'startup_health_check.json', 'coordinator.log', 'pipeline_status.json']:
        copy_file(PLAN / name, SNAPSHOT / 'experiment_plan' / name)
    copy_tree(PLAN / 'code', SNAPSHOT / 'experiment_plan/code')
    copy_tree(PLAN / 'targets', SNAPSHOT / 'experiment_plan/targets')
    for method, model in manifest['models'].items():
        for name in ['hydra.yaml', 'protocol.json', 'adapter_sources.json', 'training_complete.json', 'validation.jsonl', 'tokenizer_complete.json']:
            path = Path(model['checkpoint']) / name
            if path.exists():
                copy_file(path, SNAPSHOT / 'training_metadata' / method / name)
    for path in manifest['runtime_dependency_sha256']:
        copy_file(path, SNAPSHOT / 'runtime_source' / Path(path).relative_to(manifest['root']))
    copy_file(__file__, SNAPSHOT / 'audit_and_package.py')
    save(SNAPSHOT / 'audit_report.json', audit)
    save(SNAPSHOT / 'results.json', dict(task='PushObj', planner='CEM', seed=2000, samples_per_shape=50,
                                       training_shapes=['T'], total_episodes=700, methods=methods))
    for name, rows in [('results_by_shape.csv', table_rows), ('episodes.csv', episode_rows)]:
        with (SNAPSHOT / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n');writer.writeheader();writer.writerows(rows)
    table = '\n'.join(f"| {b['shape']} | {100 * methods['dreamerv3']['by_shape'][b['shape']]['success_rate']:.0f}% | {100 * methods['iris']['by_shape'][b['shape']]['success_rate']:.0f}% |" for b in manifest['blocks'])
    readme = f'''# DreamerV3 / IRIS completed PushObj evaluation

All **700 episodes** completed successfully as evaluation jobs (350 per method). Model-success failures remain in the denominator. Independent checks passed for all 14 shape/method groups, 700 episode files, final states, action counts, logs, exact targets, and frozen weights. No new training was run for this evaluation.

| Shape (50 episodes each) | DreamerV3 + CEM | IRIS + CEM |
|---|---:|---:|
{table}
| All seven shapes | **7.14% (25/350)** | **2.86% (10/350)** |
| Six shapes unseen by these T-only weights | 5.33% (16/300) | 2.33% (7/300) |
| I / square / small_tee comparison group | 5.33% (8/150) | 2.67% (4/150) |

## Exact evaluation protocol

- One fixed seed scheme: H7t3 base seed 2000, 50 episodes per shape. Shared original initial/goal states and cached images for both methods. Environment seed `1 + 2000 * sample_index`; CEM RNG `2000 + sample_index`, preserving the original within-shape index across GPUs.
- CEM: 200 candidates, top30, 10 iterations; horizon 5 model slots = 25 physical actions. Execute 5 physical actions, observe again and replan. At most 30 MPC replans / 150 physical actions, stop normally on success.
- Original adapter objective: decoded RGB MSE in [-1,1] plus normalized proprio MSE; terminal for the first five replans, weighted trajectory thereafter. Single-observation initialization; no CEM action clipping; model action encoding is tanh of physical relative actions.
- Success: joint agent/object position distance `norm(goal[:4] - final[:4]) < 20` and shortest-arc object angle error `< pi/9`, checked after each five-action block. Success was independently recomputed from saved final states for this archive.
- Steps use all episodes, counting failures at the full 30 MPC / 150 physical-action budget. Actual per-episode counts are explicit in NPZ and CSV files.
- GPU0 evaluated all DreamerV3 shapes; GPU1-7 evaluated one IRIS shape each. All workers loaded identical frozen tensors per method, verified by before/after and cross-worker fingerprints. Overall wall time: {elapsed / 3600:.2f} hours. DreamerV3 completed in approximately 46 minutes.

## Checkpoint coverage and adaptation

Both existing checkpoints were trained on **clean T-only `pushobj_T16k_bias`**: 16,000 training trajectories / 1,000 validation trajectories. Clean refers to visual conditions; the existing action collection retains its contact bias. These are not the newer unbiased multi-shape random16k weights. **T is training-seen; the other six shapes are unseen for these two checkpoints.**

DreamerV3 world model is epoch 3 (135,012,103 parameters). IRIS tokenizer and world model are **each** epoch 3, not three epochs in total; tokenizer has 4,027,523 trainable parameters (18,743,683 including training LPIPS), and world model has 8,517,129 total parameters. Each training stage used four-frame windows at stride five; 1,633,205 available training windows per epoch. Full training metadata, upstream commits and weight SHA256 values are included.

These are official world-model architectures with the project's offline continuous-control/proprio/MPC adaptations. There is no reward supervision, actor-critic, online RL or model update at deployment. IRIS uses a continuous macro-action/proprio embedding, an added proprio head and argmax visual-token rollout. They must not be described as complete unmodified DreamerV3/IRIS RL agents or strict original-paper reproductions.

The staged/5/30 controller is inherited from these adapters' existing evaluations. It differs from the newer random-data DINO-WM last/25/6 controller; this difference must remain disclosed. Existing PushT results are not replaced by this PushObj evaluation.

## Archive contents

- `results.json`, `results_by_shape.csv`: recomputed aggregate results and MPC/physical-action means.
- `episodes.csv`: all 700 outcomes, indices, seeds, action counts and final errors.
- `completed_runs/`: exact final NPZs, all 700 episode NPZs, per-MPC logs, configurations, frozen checks, worker logs and automatic summaries.
- `experiment_plan/`: frozen evaluation code, launch commands, authorization/protocol manifest, target caches and preflight records. Manifest status describes the prelaunch snapshot; completed status is in `completed_runs/pipeline_status.json`.
- `training_metadata/`: existing checkpoint configurations, completion records and validation metrics; weight files are not stored in Git.
- `runtime_source/`: verified simulator/wrapper/objective dependencies used in this run. The companion source tree supplies the remaining project modules; upstream model repositories and local checkpoint files must be available to rerun.
- `audit_report.json`: independent completion/integrity checks; `manifest.json` records copied-file provenance; `SHA256SUMS` covers all archived files except itself.

Absolute paths in configuration files are provenance and require remapping on another machine. This archive contains only the requested DreamerV3/IRIS evaluation, not unpublished LeWM training artifacts.
'''
    (SNAPSHOT / 'README.md').write_text(readme)
    save(SNAPSHOT / 'manifest.json', dict(created_at_UTC=datetime.now(timezone.utc).isoformat(),
        repository_relative_root=RELATIVE_ROOT, total_episodes=700, audit_passed=True,
        checkpoint_files_included=False, new_lewm_content_included=False, source_files=copied))
    files = sorted(f for f in SNAPSHOT.rglob('*') if f.is_file())
    (SNAPSHOT / 'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(SNAPSHOT)}\n' for f in files))
    print(json.dumps(dict(audit='passed', episodes=700, blocks=14, files=len(files) + 1,
        size_bytes=sum(f.stat().st_size for f in SNAPSHOT.rglob('*') if f.is_file()), methods=methods,
        archive=str(SNAPSHOT), repository_relative_root=RELATIVE_ROOT), indent=2))


if __name__ == '__main__':
    main()
