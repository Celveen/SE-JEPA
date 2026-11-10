"""CPU-only static/data validation. Never imports the simulator or model scorers."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import cv2
import numpy as np

PLAN = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = json.loads((PLAN / 'manifest.json').read_text())
    root = Path(manifest['root'])
    original = root / 'scripts/offline_visual'
    for path in [PLAN / 'run.py', PLAN / 'run_all.py', *sorted((PLAN / 'code').glob('*.py'))]:
        compile(path.read_text(), str(path), 'exec')
    for name, expected in manifest['prepared_code_sha256'].items():
        assert sha(PLAN / name) == expected, name
    for path, expected in manifest['runtime_dependency_sha256'].items():
        assert sha(path) == expected, path
    for name in ['dreamer_model.py', 'iris_model.py']:
        assert (PLAN / 'code' / name).read_bytes() == (original / name).read_bytes()
    def cem_ast(path):
        return ast.dump(next(node for node in ast.parse(path.read_text()).body
                             if isinstance(node, ast.FunctionDef) and node.name == 'cem'))
    assert cem_ast(PLAN / 'code/evaluate.py') == cem_ast(original / 'evaluate.py')
    checked_weights = {}
    for method, model in manifest['models'].items():
        folder = Path(model['checkpoint'])
        for name, expected in model['checkpoint_sha256'].items():
            assert sha(folder / name) == expected, name
        completed = json.loads((folder / 'training_complete.json').read_text())
        assert completed['epoch'] == 3
        checked_weights[method] = {'epoch': 3, 'sha256': model['checkpoint_sha256']}
    # Load only the shared constants; this module does not import simulators/models.
    constants = {}
    exec(compile((PLAN / 'code/common.py').read_text(), 'common.py', 'exec'), constants)
    rows = []
    for block in manifest['blocks']:
        assert block['n'] == 50 and block['seed'] == 2000
        for key, digest in [('target', 'target_sha256'), ('source', 'source_sha256'), ('segment_source', 'segment_sha256')]:
            assert sha(block[key]) == block[digest], block[key]
        with Path(block['source']).open('rb') as f:
            source = pickle.load(f)
        with Path(block['segment_source']).open('rb') as f:
            segments = pickle.load(f)['segments'][:50]
        with np.load(block['target'], allow_pickle=False) as target, np.load(Path(block['source']).parent / 'final_eval_results.npz') as reference:
            for key in ['state_0', 'state_g']:
                np.testing.assert_array_equal(target[key], source[key])
                np.testing.assert_array_equal(target[key], reference[key])
            np.testing.assert_array_equal(target['seeds'], 1 + 2000 * np.arange(50))
            for sample, segment in enumerate(segments):
                initial = np.r_[np.asarray(segment['states'][0], dtype=np.float64)[:5], 0., 0.]
                np.testing.assert_array_equal(target['state_0'][sample], initial)
            for obs, image, proprio in [('obs_0', 'image0', 'proprio0'), ('obs_g', 'image_g', 'proprio_g')]:
                expected_image = np.stack([cv2.resize(np.asarray(x[0]), (64, 64), interpolation=cv2.INTER_AREA) for x in source[obs]['visual']])
                expected_proprio = (np.asarray(source[obs]['proprio'])[:, 0] - constants['PROPRIO_MEAN']) / constants['PROPRIO_STD']
                np.testing.assert_array_equal(target[image], expected_image)
                np.testing.assert_allclose(target[proprio], expected_proprio, rtol=1e-6, atol=1e-6)
                assert target[image].shape == (50, 64, 64, 3) and target[image].dtype == np.uint8
                assert target[proprio].shape == (50, 4) and np.isfinite(target[proprio]).all()
        rows.append({'shape': block['shape'], 'episodes': 50, 'state_pairs_exactly_match_reference': True,
                     'cached_images_and_proprio_match': True, 'seed_mapping_match': True})
    assert len(rows) == 7 and sum(row['episodes'] for row in rows) == 350
    assert manifest['total_episodes_per_method'] == 350 and manifest['total_episodes'] == 700
    assert manifest['CEM'] == dict(samples=200, topk=30, iterations=10, model_horizon=5)
    assert manifest['max_replans'] * manifest['physical_actions_per_replan'] == 150
    python = '/data/zjc/envs/cljepa/bin/python'
    dry_runs = {}
    for method, gpu in [('dreamerv3', '0'), ('iris', '1')]:
        result = subprocess.run([python, str(PLAN / 'run.py'), method, '--gpu', gpu], check=True, capture_output=True, text=True)
        dry_runs[method] = json.loads(result.stdout)
        assert dry_runs[method]['dry_run'] and not dry_runs[method]['training']
    coordinator = subprocess.run([python, str(PLAN / 'run_all.py')], check=True, capture_output=True, text=True)
    parallel = json.loads(coordinator.stdout)
    assert parallel['dry_run'] and parallel['total_episodes'] == 700 and not parallel['training']
    jobs = parallel['jobs']
    assert len(jobs) == 8 and len({j['gpu'] for j in jobs}) == 8
    assert jobs[0]['method'] == 'dreamerv3' and jobs[0]['gpu'] == '0'
    assert jobs[0]['domains'] == [b['tag'] for b in manifest['blocks']]
    iris_tags = [tag for j in jobs[1:] for tag in j['domains']]
    assert iris_tags == [b['tag'] for b in manifest['blocks']] and len(set(iris_tags)) == 7
    for job in jobs[1:]:
        assert job['method'] == 'iris' and len(job['domains']) == 1
        result = subprocess.run([python, str(PLAN / 'run.py'), 'iris', '--gpu', job['gpu'],
                                 '--domains', *job['domains']], check=True, capture_output=True, text=True)
        worker = json.loads(result.stdout)
        assert worker['domains'] == job['domains'] and worker['gpu'] == job['gpu']
        dry_runs[job['id']] = worker
    assert not Path(manifest['output_root']).exists(), 'Unexpected new experiment output; inspect status'
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'checks': ['syntax', 'identical CEM AST', 'identical model scorer sources', 'source and target hashes',
                         'checkpoint hashes and completed epochs', 'all 350 exact state pairs',
                         'cached images/proprio', 'fixed seed mapping', 'launcher dry runs',
                         'eight-GPU schedule; each of 700 episodes assigned exactly once', 'no output directory'],
              'weights': checked_weights, 'blocks': rows, 'dry_runs': dry_runs, 'parallel_plan': parallel,
              'manifest_sha256': sha(PLAN / 'manifest.json'), 'simulation_run': False,
              'model_loaded': False, 'experiment_started': False,
              'remaining': 'User authorized launch. Next: actual GPU/inference/runtime validation and evaluation.'}
    (PLAN / 'preflight_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': True, 'episodes_per_method': 350, 'simulation_run': False,
                      'model_loaded': False, 'report': str(PLAN / 'preflight_validation.json')}, indent=2))


if __name__ == '__main__':
    main()
