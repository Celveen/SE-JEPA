"""Audit the supplied GD ablations and generate tables without pooling chains."""
from pathlib import Path
from collections import defaultdict
from itertools import product
import argparse
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/gd_ablation_20261110'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def import_source(source):
    import numpy as np
    DATA.mkdir(exist_ok=True)
    copies = {
        'README.txt': 'source_readme.txt',
        'verification.json': 'source_verification.json',
        'data/GD汇总记录.json': 'source_summary.json',
        'data/shape/shape_test_episodes.csv': 'shape_episodes.csv',
        'data/visual/visual_episodes.csv': 'visual_episodes.csv',
        'data/shape/shape_sources.json': 'shape_sources.json',
        'data/visual/visual_sources.json': 'visual_sources.json',
        'data/visual/paired_input_verification.json': 'source_paired_input_verification.json',
    }
    source_hashes = {}
    expected_hashes = dict(line.rstrip('\r\n').split('  ', 1)[::-1]
                           for line in (source / 'checksums.sha256').read_text().splitlines())
    for src, dst in copies.items():
        path = source / src
        assert sha(path) == expected_hashes[src], src
        shutil.copyfile(path, DATA / dst)
        source_hashes[dst] = sha(path)
    raw_hashes, paired = {}, {}
    for family in ('shape', 'visual'):
        groups = defaultdict(list)
        for row in rows(DATA / f'{family}_episodes.csv'):
            if family == 'shape' and row['source_kind'] != 'original_npz':
                continue
            groups[row.get('source_npz', row.get('source_path'))].append(row)
        for rel, batch in groups.items():
            path = source / 'data' / family / rel
            digest = sha(path)
            assert digest == expected_hashes[path.relative_to(source).as_posix()]
            raw_hashes[f'{family}/{rel}'] = digest
            with np.load(path, allow_pickle=False) as z:
                success = z['successes'].astype(bool).reshape(-1)
                steps = np.where(success, z['action_len'].reshape(-1), 30)
                assert len(success) == len(batch) == 50
                for row in batch:
                    i = int(row['episode_index'])
                    assert int(success[i]) == int(row['success'])
                    assert abs(float(steps[i]) - float(row['mpc_steps'])) < 1e-10
                if family == 'visual':
                    key = (batch[0]['domain'], batch[0]['test_seed'])
                    hashes = {k: hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()
                              for k in ('state_0', 'state_g')}
                    assert paired.setdefault(key, hashes) == hashes
    assert len(raw_hashes) == 966 and len(paired) == 21
    report = dict(source_folder=source.name, source_files_sha256=source_hashes,
                  raw_npz_checked=len(raw_hashes), raw_npz_sha256=raw_hashes,
                  visual_matched_initial_goal_groups=len(paired),
                  shape_state_pairing_limit='Historical full-model and official-reference CSVs lack original state tensors; no independent physical-state pairing claim is made from these CSVs.')
    (DATA / 'source_audit.json').write_text(json.dumps(report, indent=2) + '\n')


def recompute():
    groups = defaultdict(list)
    keys = set()
    for family in ('shape', 'visual'):
        for row in rows(DATA / f'{family}_episodes.csv'):
            assert row.get('optimizer', 'GD') == 'GD'
            method, chain = row['method'], int(row['chain'])
            key = family, method, chain, row['domain'], int(row['test_seed']), int(row['episode_index'])
            assert key not in keys
            keys.add(key)
            s, t = int(row['success']), float(row['mpc_steps'])
            assert s in (0, 1) and 0 <= t <= 30 and (s or t == 30)
            groups[family, method, chain].append(row)
    reference = json.loads((DATA / 'source_summary.json').read_text())['records']
    out = {}
    for key, batch in groups.items():
        assert len(batch) == 1050
        domains = {r['domain'] for r in batch}
        assert len(domains) == 7
        assert {(r['domain'], int(r['test_seed']), int(r['episode_index'])) for r in batch} == set(product(domains, (2000,2100,2200), range(50)))
        count = sum(int(r['success']) for r in batch)
        total = sum(float(r['mpc_steps']) for r in batch)
        out[key] = dict(n=len(batch), successes=count, success_pct=100*count/len(batch), replans=total/len(batch))
        target = [r for r in reference if (r['domain'], r['method'], r['chain'] or 0) == key]
        assert len(target) == 1
        assert target[0]['n'] == len(batch) and target[0]['successes'] == count
        assert abs(target[0]['mpc_all'] - total/len(batch)) < 1e-9
    assert len(out) == 51 and len(keys) == 53550
    for target in reference:
        if target['aggregation'] != 'three_chain_pooled':
            continue
        batch = [out[target['domain'], target['method'], c] for c in (1,2,3)]
        assert sum(r['n'] for r in batch) == target['n']
        assert sum(r['successes'] for r in batch) == target['successes']
        assert abs(sum(r['replans'] for r in batch)/3 - target['mpc_all']) < 1e-9
    return out


COMPONENTS = [
    (r'Full \ours', 'ours_lr1e4'),
    ('Episodic reset', 'adajepa_same'),
    ('Inherit fast weights', 'naive'),
    ('Shared predictor: EMA', 'ema_slow'),
    ('Shared predictor: replay', 'shared_slow'),
    ('No maturity screening', 'no_maturity_read'),
    ('No familiar-read protection', 'no_familiar_read'),
    ('No familiar-write protection', 'no_familiar_write'),
    ('Neither familiarity protection', 'no_familiarity'),
]
RATES = [(r'$5\times10^{-5}$','slow_lr_5e5'),
         (r'$10^{-4}$ (default)','ours_lr1e4'),
         (r'$2\times10^{-4}$','slow_lr_2e4'),
         (r'$3\times10^{-4}$','slow_lr_3e4'),
         (r'$5\times10^{-4}$','slow_lr_5e4')]


def generate(out):
    def cell(family, method, chain, with_replans=False):
        if method == 'adajepa_same' and family == 'visual':
            chain = 0
        r = out.get((family, method, chain))
        if r is None:
            return r'\missing'
        return f"{r['success_pct']:.2f}" + (f" / {r['replans']:.2f}" if with_replans else '')
    for filename, detailed in [('gd_ablation_main.tex', False), ('gd_ablation_detailed.tex', True)]:
        lines = []
        columns = [('shape',1), ('visual',1), ('visual',2)]
        if detailed:
            columns.append(('visual',3))
        for group_i, group in enumerate((COMPONENTS, RATES)):
            if group_i:
                lines += [r'\midrule', r'\multicolumn{' + str(len(columns)+1) + r'}{@{}l}{Full \ours: fixed slow learning rate}\\']
            for label, method in group:
                lines.append(' & '.join([label] + [cell(f,method,c,detailed) for f,c in columns]) + r' \\')
        (ROOT/'tables'/filename).write_text('\n'.join(lines)+'\n')
    pooled = []
    for family in ('shape','visual'):
        for method in ('ours_lr1e4','naive'):
            vals = [out[family,method,c] for c in (1,2,3)]
            label = (r'Full \ours' if method=='ours_lr1e4' else 'Inherit fast weights')
            pooled.append(' & '.join(['PushObj' if family=='shape' else 'PushT', label] + [f"{v['success_pct']:.2f}" for v in vals] + [f"{sum(v['success_pct'] for v in vals)/3:.2f}", f"{sum(v['replans'] for v in vals)/3:.2f}"]) + r' \\')
    (ROOT/'tables/gd_inheritance_chains.tex').write_text('\n'.join(pooled)+'\n')
    audit = dict(unique_conditions=len(out), unique_evaluations=sum(v['n'] for v in out.values()),
                 all_source_records_recomputed=True, failures_count_as_replans=30,
                 heldout_evaluations_per_chain=1050, test_seeds=[2000,2100,2200],
                 visual_episodic_reset_reused_not_independent=True,
                 raw_source_audit='source_audit.json',
                 results=[dict(family=k[0],method=k[1],chain=k[2],**v) for k,v in sorted(out.items())])
    (DATA/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(f'Verified {len(out)} GD conditions and {audit["unique_evaluations"]} evaluations; wrote three tables.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.source:
        import_source(args.source)
    generate(recompute())
