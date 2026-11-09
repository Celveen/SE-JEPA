from pathlib import Path
from collections import defaultdict
import json, hashlib, statistics, csv
import numpy as np

import argparse
parser = argparse.ArgumentParser(description="Verify the supplied mixed-14 archive and recompute its statistics")
parser.add_argument("archive_root", type=Path)
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/mixed14_audit.json")
args = parser.parse_args()
BASE = args.archive_root.resolve()
args.output.parent.mkdir(parents=True, exist_ok=True)
docs=[json.loads((BASE/f'analysis/mixed14_{name}_final_results.json').read_text()) for name in ['ours','baseline']]
records=[]
checks=defaultdict(int)
targets={}
checkpoints={}
for doc in docs:
    protocol=doc['protocol']; root=BASE/'raw_data'/protocol['name']
    state_by_chain={}; hash_by_chain={}
    for r in sorted(doc['results'],key=lambda x:(x.get('repetition',1),x['optimizer'],x['index'])):
        r=dict(r)
        path=root/'results'/r['path'].split('/results/',1)[1]
        p=path/'final_eval_results.npz'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['artifact_sha256']['final_eval_results.npz']
        with np.load(p,allow_pickle=False) as z:
            success=z['successes'].astype(bool)
            steps=z['action_len']
            assert len(success)==r['n']==50
            assert int(success.sum())==r['successes']
            assert [int((success & (steps<=k)).sum()) for k in range(1,31)]==r['success_counts_by_mpc_step']
            assert np.all(np.isinf(steps[~success]) | (steps[~success]==30))
            r['verified_replans_sum']=float(np.where(success,steps,30).sum())
            target=hashlib.sha256(z['state_0'].tobytes()+z['state_g'].tobytes()).hexdigest()
            if r['tag'] in targets:assert targets[r['tag']]==target
            targets[r['tag']]=target
            with np.load(root/'target_references'/(r['tag']+'.npz'),allow_pickle=False) as ref:
                for k in ['state_0','state_g']:assert np.array_equal(z[k],ref[k])
        if r['method']=='v6a':
            key=(r['repetition'],r['optimizer'])
            a=path/'state_load_audit.json'
            assert hashlib.sha256(a.read_bytes()).hexdigest()==r['artifact_sha256'][a.name]
            a=json.loads(a.read_text())
            assert a['initial']['episode']==r['initial_episode']==r['index']*50
            assert a['final']['episode']==r['final_episode']==(r['index']+1)*50
            if r['index']==0:
                assert a['initial']==dict(episode=0,pending_n=0,domains={})
                assert r['state_in'] is None
                checks['empty_chain_starts']+=1
            else:
                assert a['initial']==state_by_chain[key]
                assert r['state_in_sha256']==hash_by_chain[key]
                checks['consecutive_state_hash_links']+=1
            state_by_chain[key]=a['final']; hash_by_chain[key]=r['state_sha256']
            checks['state_provenance_files']+=1
        checks['npz_files']+=1;checks['episodes']+=r['n'];checks['successes']+=r['successes']
        records.append(r)

def stats(rows):
    units=defaultdict(list)
    for r in rows:units[r.get('repetition',r['seed'])].append(r)
    rates=[100*sum(x['successes'] for x in rs)/sum(x['n'] for x in rs) for rs in units.values()]
    replans=[sum(x['verified_replans_sum'] for x in rs)/sum(x['n'] for x in rs) for rs in units.values()]
    return dict(episodes=sum(r['n'] for r in rows),successes=sum(r['successes'] for r in rows),mean_pct=statistics.mean(rates),sd_pct=statistics.stdev(rates),unit_rates_pct=rates,mean_replans=statistics.mean(replans),sd_replans=statistics.stdev(replans))

rounds={opt:{family:[stats([r for r in records if r['method']=='v6a' and r['optimizer']==opt and r['round']==rd and (family=='all' or r['family']==family)]) for rd in range(1,6)] for family in ['all','visual','shape']} for opt in ['GD','CEM']}
table=[]
for opt in ['GD','CEM']:
    for method in ['frozen','adajepa','v6a']:
        rr=[r for r in records if r['method']==method and r['optimizer']==opt and (method!='v6a' or r['round']==5)]
        table.append(dict(planner=opt,method=method,**{fam:stats([r for r in rr if fam=='all' or r['family']==fam]) for fam in ['all','visual','shape']}))
domain_table={opt:{tag:{method:stats([r for r in records if r['optimizer']==opt and r['tag']==tag and r['method']==method and (method!='v6a' or r['round']==5)]) for method in ['frozen','adajepa','v6a']} for tag in docs[0]['protocol']['domain_order']} for opt in ['GD','CEM']}
old=json.loads((BASE/'analysis/mixed14_summary.json').read_text())
for row in table:
    for family in ['all','visual','shape']:
        expected=old['summary']['final'][row['method']][row['planner']][family]
        assert expected['episodes']==row[family]['episodes'] and expected['successes']==row[family]['successes']
        assert abs(expected['rate_pct']-row[family]['mean_pct'])<1e-10
        assert abs(expected['sd_pct']-row[family]['sd_pct'])<1e-10
for row in csv.DictReader((BASE/'figures/十四域_8页/data/MPC平均步数数据.csv').read_text(encoding='utf-8-sig').splitlines()):
    method={'Frozen':'frozen','AdaJEPA':'adajepa','Ours':'v6a'}[row['method']]
    actual=next(x['all'] for x in table if x['planner']==row['planner'] and x['method']==method)
    assert abs(float(row['all_mean'])-actual['mean_replans'])<1e-10
    assert abs(float(row['all_sd'])-actual['sd_replans'])<1e-10

output={
 'source_root':str(BASE), 'source_pdf':str(BASE/'figures/十四域_8页/十四域_图表合集.pdf'),
 'status':'raw_outcomes_verified', 'independent_validation':dict(checks),
 'protocol':{k:docs[0]['protocol'][k] for k in ['domain_order','round_domain_orders','target_pairing','state_policy','seed_schedule','episodes_per_domain_per_round','rounds','repetitions','checkpoint_description']},
 'statistical_unit':'Sample SD over 3 independently initialized continual chains, or 3 baseline random seeds. Fixed target pairs are reused across all rounds and chains. No held-out evaluation.',
 'checkpoint_provenance':json.loads((BASE/'weights/base_checkpoint_requirements.json').read_text()),
 'round_curves':rounds,'final_comparison':table,'final_by_domain':domain_table,
 'page_mapping':{'final_comparison':1,'round1_to3_pooled':2,'final_per_domain':3,'five_round_curves':4,'replans_summary':5,'improvement_vs_baselines':6,'replans_per_domain':7,'replans_table':8},
 'supported_claims':[
  'One continual memory is carried across all seven visual and seven shape domains for five rounds; six chains start empty, 414 consecutive state hashes and 420 state audits match.',
  'Overall online success improves monotonically in the three-chain aggregate across all five rounds for GD and CEM.',
  'Both geometric-shift and visual-shift families improve from the first to fifth round.',
  'AdaJEPA and Frozen use the official pushobj_shape_shift checkpoint in this experiment, including its visual-domain blocks.'
 ],
 'claim_limits':[
  'Targets are repeated, not new held-out targets; the seeds vary run randomness and do not imply independent goal draws.',
  'This is a 7+7 mixed stream, not all 7x7 combinations of object geometry and visual conditions.',
  'One state persists across both families, but no isolated cross-family transfer control establishes causal positive transfer.',
  'All domains remain in the planar pushing simulator; prefer cross-domain or mixed geometric-and-visual-shift learning over broad multi-task generalization.',
  'Method initialization and adaptation exposure differ: SE-JEPA uses the custom TLZ+ epoch-3 model and persistent learning; baselines use official shape-shift weights and reset per episode.',
  'Aggregate five-round improvement does not mean every individual chain improves at every round, nor that every domain exceeds AdaJEPA.'
 ],
 'proposed_main_text':'A single continual memory also spans a mixed stream of seven visual conditions and seven object geometries. Across three five-round chains, online success rises from 65.6% to 75.1% with GD and from 65.0% to 75.8% with CEM, with gains in both domain families. This repeated-target experiment demonstrates continued improvement under alternating geometric and visual shifts.',
 'short_protocol_note':'Three independent continual chains revisit fixed targets over five rounds (50 episodes per domain per round). SE-JEPA starts from our TLZ+ model; AdaJEPA and Frozen use the official shape-shift checkpoint. Round-5 results are online; initialization and adaptation exposure differ.'
}
source_names=['figures/十四域_8页/十四域_图表合集.pdf','analysis/mixed14_summary.json','analysis/mixed14_ours_final_results.json','analysis/mixed14_baseline_final_results.json','raw_data/mixed_cls6a_14domains_roundseeds100_500_20261024/final_results.json','raw_data/mixed_official_shape_baselines_3seeds_20261024/final_results.json','verification/formal_batches.json','figures/十四域_8页/English_captions.txt','weights/base_checkpoint_requirements.json']
output['source_files']=[dict(path=str(BASE/f),relative_path=f,bytes=(BASE/f).stat().st_size,sha256=hashlib.sha256((BASE/f).read_bytes()).hexdigest()) for f in source_names]
code=Path(__file__)
output['verification_code']=dict(path=str(code),sha256=hashlib.sha256(code.read_bytes()).hexdigest(),bytes=code.stat().st_size)
output['domain_comparison_summary']={}
for opt,dd in domain_table.items():
    delta={tag:v['v6a']['mean_pct']-v['adajepa']['mean_pct'] for tag,v in dd.items()}
    gain=[k for k,v in delta.items() if v>1e-9]
    tie=[k for k,v in delta.items() if abs(v)<1e-9]
    loss=[k for k,v in delta.items() if v<-1e-9]
    output['domain_comparison_summary'][opt]=dict(wins=len(gain),ties=len(tie),losses=len(loss),winning_domains=gain,tied_domains=tie,lower_domains=loss,delta_pp=delta,largest_gain_domain=max(delta,key=delta.get),red_object_gain_pp=delta['red_object'])
out=args.output;out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(out),'checks':checks,'round_curves':{o:{f:[round(x['mean_pct'],3) for x in v] for f,v in d.items()} for o,d in rounds.items()}},ensure_ascii=False))
