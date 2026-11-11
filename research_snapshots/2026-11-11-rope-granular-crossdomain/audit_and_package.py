"""Audit and archive completed mixed-domain continual-learning results; no experiment execution."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import re
import shutil

import numpy as np
from omegaconf import OmegaConf

HERE=Path(__file__).resolve().parent
WORK=Path('/data/zjc/adaJEPA')
ROOT=WORK/'CL-JEPA-cudagraph'
SNAPSHOT=HERE/'snapshot'
RELATIVE='research_snapshots/2026-11-11-rope-granular-crossdomain'
SPECS=[('lr2e4_with_familiarity','rope_granular_lr2e4_mixed_20261103',3,2e-4,True),
       ('lr1e4_with_familiarity','rope_granular_lr1e4_mixed_20261104',1,1e-4,True),
       ('lr1e4_no_familiarity','rope_granular_no_familiarity_lr1e4_mixed_20261106',3,1e-4,False)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    assert not SNAPSHOT.exists(),'Existing archive; inspect before replacing'
    SNAPSHOT.mkdir();files=[];episodes=[];round_rows=[];run_rows=[];variants={};targets={}
    old_index=list(csv.DictReader((ROOT/'research_snapshots/2026-11-07/run_index.csv').open()))
    def store(source,destination,compress=False):
        source=Path(source);destination=SNAPSHOT/destination;destination.parent.mkdir(parents=True,exist_ok=True)
        if compress:
            destination=destination.with_name(destination.name+'.gz')
            content=source.read_bytes();destination.write_bytes(gzip.compress(content,mtime=0))
            assert gzip.decompress(destination.read_bytes())==content
        else:shutil.copy2(source,destination)
        files.append(dict(source=str(source),source_sha256=sha(source),stored=str(destination.relative_to(SNAPSHOT)),
                          stored_sha256=sha(destination),bytes=destination.stat().st_size,lossless_gzip=compress))
    for tag,name,rounds,lr,familiar in SPECS:
        plan=WORK/'experiment_plans'/name;m=json.loads((plan/'manifest.json').read_text());output=Path(m['output'])
        old_runs=[r for r in old_index if r['group']=='CL-JEPA-cudagraph/'+name]
        previous_chunks={Path(r['source']).name for r in old_runs};completed=[];new_chunks=[]
        performance=defaultdict(list);final_domains={}
        for block in m['blocks']:
            folder=output/'free_abs'/block['chunk']
            if block['round']>rounds:
                assert not (folder/'final_eval_results.npz').exists(),'Unexpected expanded experiment'
                continue
            cfg=OmegaConf.load(folder/'.hydra/config.yaml')
            assert cfg.n_evals==10 and cfg.seed==block['seed'] and cfg.goal_source=='file'
            assert cfg.planner.max_iter==3 and cfg.planner.n_taken_actions==5
            assert cfg.planner.sub_planner.horizon==5 and cfg.planner.sub_planner.num_samples==300
            assert cfg.planner.sub_planner.topk==30 and cfg.planner.sub_planner.opt_steps==10
            assert cfg.planner.continual.slow_lr==lr
            assert str(cfg.ckpt_base_path).endswith('rope_dino_h1_fs1_e100_seed0')
            if not familiar:
                assert cfg.planner.continual.familiar_thr==0 and cfg.planner.continual.familiar_read is False
            console=(folder/'console.log').read_text(errors='replace')
            assert 'Traceback (most recent call last)' not in console and 'simulator exploded' not in console
            values=np.asarray([float(x) for x in re.findall(r'^CD:\s+(\S+)',console,re.M)])
            assert values.size==30 and np.isfinite(values).all()
            by_step=values.reshape(10,3);final=by_step[:,-1]
            logs=[json.loads(s) for s in (folder/'logs.json').read_text().splitlines() if s]
            logged=[r['final_eval/mean_chamfer_distance'] for r in logs if 'final_eval/mean_chamfer_distance' in r][-1]
            assert np.isclose(final.mean(),logged,rtol=0,atol=1e-6)
            history=[json.loads(s) for s in (folder/'continual_episodes.jsonl').read_text().splitlines() if s]
            assert len(history)==10
            assert [r['episode'] for r in history]==list(range(block['index']*10,(block['index']+1)*10))
            assert all(r['tag']==block['object'] for r in history)
            if not familiar:assert all(r['write_rule']!='familiar' for r in history)
            target=Path(block['target'])
            if block['chunk'] in targets:
                assert targets[block['chunk']]['source_sha256']==sha(target)
            else:
                targets[block['chunk']]=dict(source=str(target),source_sha256=sha(target))
                store(target,Path('shared_targets')/target.name,compress=True)
                sidecar=target.with_suffix('.json')
                if sidecar.exists():store(sidecar,Path('shared_targets')/sidecar.name)
            with target.open('rb') as f:target_data=pickle.load(f)
            with np.load(folder/'final_eval_results.npz',allow_pickle=False) as data:
                assert data['actions'].shape==(10,15,4) and np.isfinite(data['actions']).all()
                for key in ['state_0','state_g']:np.testing.assert_array_equal(data[key],target_data[key])
            for sample,(value,record) in enumerate(zip(final,history)):
                origin=block['samples'][sample]
                episodes.append(dict(variant=tag,round=block['round'],object=block['object'],chunk=block['chunk'],
                    global_episode=record['episode'],sample=sample,target_trajectory=origin['trajectory'],target_offset=origin['offset'],
                    environment_seed=origin['env_seed'],final_CD=float(value),CD_after_push5=float(by_step[sample,0]),
                    CD_after_push10=float(by_step[sample,1]),CD_after_push15=float(by_step[sample,2]),
                    read=str(record.get('read')),write=str(record.get('write')),write_rule=record['write_rule'],
                    final_active_domains=len([v for v in record['domains'].values() if v['n']>0])))
                performance[(block['round'],block['object'])].append(float(value))
            final_domains[block['round']]=history[-1]['domains']
            destination=Path('completed_runs')/tag/block['chunk']
            for file in sorted(folder.iterdir()):
                if file.is_file() and file.name in ['final_eval_results.npz','console.log','plan.log','logs.json','per_iter_logs.json','continual_episodes.jsonl','adajepa_losses.csv','probe_meta.json']:
                    store(file,destination/file.name,compress=file.suffix in ['.log','.jsonl','.csv'] or file.name in ['logs.json','per_iter_logs.json'])
            for file in sorted((folder/'.hydra').glob('*.yaml')):store(file,destination/'.hydra'/file.name)
            np.savez_compressed(SNAPSHOT/destination/'CD_by_replan.npz',CD=by_step,final_CD=final)
            if block['chunk'] in previous_chunks:
                previous=ROOT/'research_snapshots/2026-11-07/completed_runs/CL-JEPA-cudagraph'/name/'free_abs'/block['chunk']/'final_eval_results.npz'
                assert sha(previous)==sha(folder/'final_eval_results.npz'),'Previously archived result changed'
            else:new_chunks.append(block['chunk'])
            completed.append(block['chunk'])
            run_rows.append(dict(variant=tag,round=block['round'],object=block['object'],chunk=block['chunk'],episodes=10,
                                 final_CD_mean=float(final.mean()),previously_archived=block['chunk'] in previous_chunks))
        assert len(completed)==rounds*10
        reviews=[]
        for round_id in range(1,rounds+1):
            review=json.loads((plan/f'round_{round_id}_review.json').read_text())
            for obj in ['rope','granular']:
                values=np.asarray(performance[round_id,obj]);assert len(values)==50
                assert np.isclose(values.mean(),review['performance'][obj]['final_cd_mean'],rtol=0,atol=1e-10)
                round_rows.append(dict(variant=tag,slow_lr=lr,familiarity_gate=familiar,round=round_id,object=obj,
                    episodes=50,final_CD_mean=float(values.mean()),final_CD_median=float(np.median(values))))
            assert final_domains[round_id]==review['domains']
            reviews.append(dict(round=round_id,rope_CD=float(np.mean(performance[round_id,'rope'])),
                granular_CD=float(np.mean(performance[round_id,'granular'])),domains=review['domains'],
                member_weighted_purity=review['member_weighted_purity']))
        for file in sorted(plan.rglob('*')):
            if file.is_file() and file.suffix in ['.py','.md','.json','.yaml'] and '__pycache__' not in file.parts and 'targets' not in file.relative_to(plan).parts:
                store(file,Path('experiment_plans')/name/file.relative_to(plan))
        for file in sorted((output/'pipeline_logs').glob('*')):
            if file.is_file() and file.suffix in ['.log','.json','.jsonl']:store(file,Path('pipeline_logs')/tag/file.name,compress=True)
        variants[tag]=dict(source_plan=name,slow_lr=lr,familiarity_gate=familiar,completed_rounds=rounds,planned_rounds=3,
            completed_episodes=rounds*100,status='completed' if rounds==3 else 'stopped_after_round1_not_complete_three_rounds',
            previously_archived_chunks=len(old_runs),newly_completed_chunks_added=new_chunks,rounds=reviews)
    assert len(episodes)==700 and len(run_rows)==70 and len(targets)==30
    for filename,rows in [('episodes.csv',episodes),('round_CD.csv',round_rows),('run_index.csv',run_rows)]:
        with (SNAPSHOT/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    summary=dict(task='Rope/Granular mixed-domain continual learning',audited_at_UTC=datetime.now(timezone.utc).isoformat(),
        primary_metric='Final particle Chamfer distance after15 pushes; lower is better',completed_episodes=700,
        no_new_experiments=True,variants=variants)
    save(SNAPSHOT/'results.json',summary)
    save(SNAPSHOT/'audit_report.json',dict(status='passed',complete_chunks=70,episodes=700,shared_target_blocks=30,
        new_result_chunks_missing_from_previous_archive=9,checks=['Finite actions and30 logged CD evaluations per chunk',
        'Per-chunk final means agree with logs; per-round50/object means agree with reviews',
        'Sequential episode IDs and object labels match routing logs', 'Exact shared initial/goal states and source hashes',
        'Previously archived61 chunk NPZs match byte for byte', 'Final routing/domain summaries agree with reviews'],
        no_model_weights_included=True,no_training_data_included=True))
    rows=[]
    for tag in variants:
        for review in variants[tag]['rounds']:
            rows.append(f"| {tag} | {review['round']} | {review['rope_CD']:.6f} | {review['granular_CD']:.6f} |")
    readme='''# Rope / Granular mixed-domain continual-learning results — complete archive

This archive completes the previously partial mixed-domain result publication and consolidates the matching learning-rate/familiarity comparisons. All70 completed ten-episode chunks (700 episodes across the three variants) were audited. No experiment, training, seed search or replanning was run to publish these files.

## What was missing

The previous `baselines` archive `research_snapshots/2026-11-07` contains the full2e-4 three-round chain (300 episodes), the stopped1e-4 with-familiarity first round (100), and only the first21/30 chunks of the1e-4 no-familiarity chain (210/300). The last9 chunks, final third-round review and completed pipeline record were absent. This directory includes all300 episodes and final domain diagnostics for that latest chain. The700 total here includes the400 already completed comparison episodes; it is not700 newly run episodes.

## Final CD by round

Each completed round has50 Rope and50 Granular tasks. The two3-round streams use different pre-generated target groups each round; all three variants share the same matching target set/seed where they overlap.

| Variant | Round | Rope CD | Granular CD |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

`lr1e4_no_familiarity` is the latest user-requested chain: slow lr1e-4, `familiar_read=false`, `familiar_thr=0`. Read selection compares the base and eligible mature learners by relative one-step latent prediction error; it does not use true object labels to select a learner. Maturity, write rejection and the other domain-management rules remain enabled. The old1e-4 with-familiarity run stopped after its first round under its historical performance gate; missing rounds are not fabricated or silently rerun.

## Domain separation in the latest chain

After round3, two mature domains exist: d0 contains148 Rope members; d1 contains150 Granular and1 Rope member. Member-weighted purity is298/299=99.6656%. Thus separation is nearly pure, not perfectly pure. Counts describe accepted domain members, not all300 evaluated episodes. CD improvement from each task's initial state is not by itself an algorithmic gain versus a separately paired frozen baseline.

## Protocol and disclosure

- Shared base: locally trained `rope_dino_h1_fs1_e100_seed0`, epoch100/latest; both objects use this Rope base. This is online cross-domain continual learning, not the later frozen ECP trials or the Granular-specific frozen checkpoint.
- Per round: `(Rope10 -> Granular10) x5`,50 tasks per object,100 total; one evolving learning chain per variant. The three target-round groups are disjoint; they are not three independent model seeds or repeated passes over the same50 targets.
- CEM300 candidates, top30,10 iterations, horizon5 pushes; MPC executes5 pushes per replan,3 replans /15 pushes, original last objective. The later horizon5/execute1 experiment does not change these historical results.
- Fast adaptation and slow learning remain active. Saved configurations include predictor adaptation lr5e-4, last-layer encoder adaptation switches/lr1e-5, and slow15 steps/batch8 per normal write. Exact configurations are provided; do not describe this as frozen inference.
- Primary publication metric is CD. Raw deformable-environment success flags are not the paper's task success metric. Historical round reviews retain their separately labelled CD-threshold success diagnostics; those are not used as the primary summary here.
- Some raw diagnostic JSON fields deliberately contain NaN/Infinity for undefined cosine/familiarity diagnostics or disabled thresholds. These are preserved exactly, not substituted for missing CD/action values. Physical CD and action finiteness were checked.
- The no-familiarity chain recovered from an I/O stall after episode190 and retained already completed files, then finished all300. Recovery records are included. No simulation failure or nonfinite CD was found in the reviewed mixed-chain outputs.

## Files

- `results.json`, `round_CD.csv`: recomputed CD summaries and domain memberships for each variant/round.
- `episodes.csv`:700 per-task CD values after5/10/15 pushes, target source IDs, seeds and read/write routing.
- `completed_runs/`: all70 exact result NPZs, derived `CD_by_replan.npz`, configurations, per-iteration/episode routing and adaptation logs, console logs. Text logs use lossless gzip.
- `shared_targets/`: all30 original shared target pickle files in lossless gzip, plus source sidecars. Decompress them before use; remap absolute paths from saved configurations.
- `experiment_plans/`, `pipeline_logs/`: source launch/analyzer code, protocol manifests, per-round reviews, launch/recovery/completion records.
- `audit_report.json`, `manifest.json`, `SHA256SUMS`: integrity checks and source/stored-file hashes. Verify with `sha256sum -c SHA256SUMS` from this directory.

Large evolving learner checkpoints and saved episode RGB caches remain on the source server and are not included in Git. They were not needed to republish the saved numerical/routing results. No unpublished LeWM artifacts or current ECP reruns are included.
'''
    (SNAPSHOT/'README.md').write_text(readme)
    store(__file__,Path('audit_and_package.py'))
    save(SNAPSHOT/'manifest.json',dict(repository_relative_root=RELATIVE,created_at_UTC=datetime.now(timezone.utc).isoformat(),
        files=files,total_completed_chunks=70,total_episodes=700,newly_missing_chunks_added=9,large_model_weights_included=False))
    allfiles=sorted(f for f in SNAPSHOT.rglob('*') if f.is_file())
    (SNAPSHOT/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(SNAPSHOT)}\n' for f in allfiles))
    print(json.dumps(dict(status='audited_and_packaged',episodes=700,chunks=70,missing_chunks_added=9,files=len(allfiles)+1,
        bytes=sum(f.stat().st_size for f in SNAPSHOT.rglob('*') if f.is_file()),snapshot=str(SNAPSHOT)),indent=2))


if __name__=='__main__':main()
