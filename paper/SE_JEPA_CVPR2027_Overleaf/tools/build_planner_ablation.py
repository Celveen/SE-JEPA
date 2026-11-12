"""Recompute GD/CEM ablations from episode records; retain complete task chains."""
from pathlib import Path
from collections import defaultdict
from itertools import product
import argparse, csv, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/cem_ablation_20261112'
FIELDS = ['optimizer', 'family', 'method', 'chain', 'domain', 'test_seed',
          'episode_index', 'success', 'mpc_steps']
COMMON = [(r'Full \ours', 'ours_lr1e4'),
          ('No maturity screening', 'no_maturity_read'),
          ('No familiarity protection', 'no_familiarity')]
RATES = [(r'$5\times10^{-5}$', 'slow_lr_5e5'),
         (r'$10^{-4}$ (default)', 'ours_lr1e4'),
         (r'$2\times10^{-4}$', 'slow_lr_2e4'),
         (r'$3\times10^{-4}$', 'slow_lr_3e4'),
         (r'$5\times10^{-4}$', 'slow_lr_5e4')]
DESIGN = [(r'Full \ours', 'ours_lr1e4'), ('Episodic reset', 'adajepa_same'),
          ('Inherit fast weights', 'naive'), ('Shared predictor: EMA', 'ema_slow'),
          ('Shared predictor: replay', 'shared_slow'),
          ('No familiar-read protection', 'no_familiar_read'),
          ('No familiar-write protection', 'no_familiar_write')]

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def import_source(source):
    DATA.mkdir(exist_ok=True)
    groups, hashes, selected = defaultdict(list), {}, []
    for family in ('shape', 'visual'):
        path = source / 'data' / family / 'episodes.csv'
        hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for r in read_csv(path):
            r['family'] = family
            groups[family, r['method'], int(r['chain'])].append(r)
            if int(r['chain']) == (1 if family == 'shape' else 2):
                selected.append({k: r[k] for k in FIELDS})
    summary = source / 'data/CEM汇总记录.json'
    hashes[str(summary.relative_to(source))] = hashlib.sha256(summary.read_bytes()).hexdigest()
    records = json.loads(summary.read_text())['records']
    assert len(groups) == len(records) == 20
    for r in records:
        batch = groups[r['domain'], r['method'], r['chain']]
        domains = {x['domain'] for x in batch}
        keys = {(x['domain'], int(x['test_seed']), int(x['episode_index'])) for x in batch}
        assert len(batch) == len(keys) == r['n'] == 1050
        assert keys == set(product(domains, (2000, 2100, 2200), range(50)))
        assert sum(int(x['success']) for x in batch) == r['successes']
        assert abs(sum(float(x['mpc_steps']) for x in batch)/1050-r['mpc_all']) < 1e-9
        assert all(int(x['success']) or float(x['mpc_steps']) == 30 for x in batch)
    with (DATA/'episodes.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(selected)
    (DATA/'source_audit.json').write_text(json.dumps(dict(
        source_folder=source.name, source_sha256=hashes, verified_conditions=20,
        verified_evaluations=21000, retained_evaluations=len(selected),
        selected_chains={'PushObj':1,'PushT':2},
        selection_reason='Complete common ablation grid; matches the retained GD task chains.'), indent=2)+'\n')

def generate():
    from build_gd_ablation import recompute
    out = {('GD',)+k:v for k,v in recompute().items()}
    groups = defaultdict(list)
    for r in read_csv(DATA/'episodes.csv'):
        groups['CEM',r['family'],r['method'],int(r['chain'])].append(r)
    for k,rows in groups.items():
        assert len(rows)==1050
        out[k] = dict(n=len(rows), successes=sum(int(r['success']) for r in rows),
                      success_pct=100*sum(int(r['success']) for r in rows)/len(rows),
                      replans=sum(float(r['mpc_steps']) for r in rows)/len(rows))
    columns = [(p,f,c) for f,c in [('shape',1),('visual',2)] for p in ['GD','CEM']]
    used = set()
    def cell(p,f,m,c,detailed):
        if m=='adajepa_same' and f=='visual': c=0
        used.add((p,f,m,c)); v=out[p,f,m,c]
        return f"{v['success_pct']:.2f}"+(f" / {v['replans']:.2f}" if detailed else '')
    for detailed in (False,True):
        lines=[]
        for i,group in enumerate([COMMON,RATES]):
            if i: lines += [r'\midrule',r'\multicolumn{5}{@{}l}{Full \ours: fixed slow learning rate}\\']
            for label,method in group:
                lines.append(' & '.join([label]+[cell(p,f,method,c,detailed) for p,f,c in columns])+r' \\')
        name='planner_ablation_detailed.tex' if detailed else 'planner_ablation_main.tex'
        (ROOT/'tables'/name).write_text('\n'.join(lines)+'\n')
    lines=[' & '.join([label]+[cell('GD',f,m,c,True) for f,c in [('shape',1),('visual',2)]])+r' \\' for label,m in DESIGN]
    (ROOT/'tables/gd_memory_design.tex').write_text('\n'.join(lines)+'\n')
    (DATA/'audit.json').write_text(json.dumps(dict(
        evaluations_per_cell=1050, test_seeds=[2000,2100,2200], failure_replans=30,
        results=[dict(planner=k[0],family=k[1],method=k[2],chain=k[3],**out[k]) for k in sorted(used)],
        missing_cem_variants=[m for _,m in DESIGN if m!='ours_lr1e4']),indent=2)+'\n')
    print('Verified 51 GD conditions; exported 14 complete CEM conditions and selected GD comparisons.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path); a=p.parse_args()
    if a.source: import_source(a.source)
    generate()
