"""Prepare a matched mixed stream with familiarity filtering disabled."""
import copy
import hashlib
import json
from pathlib import Path

from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

PLAN=Path(__file__).resolve().parent
REFERENCE=PLAN.parent/'rope_granular_lr1e4_mixed_20261104'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    if (PLAN/'manifest.json').exists():raise FileExistsError('Manifest already prepared')
    old=json.loads((REFERENCE/'manifest.json').read_text())
    old_jobs=json.loads((REFERENCE/'commands.json').read_text())
    m=copy.deepcopy(old);root=Path(m['root']);out=root/'eval_outputs/rope_granular_no_familiarity_lr1e4_mixed_20261106'
    if out.exists():raise FileExistsError(out)
    m.pop('first_round_policy',None)
    m.update(status='authorized_prepared',output=str(out),chain_gpu='1',reference_plan=str(REFERENCE),
        familiar_thr=0.,familiar_read=False,
        continuation_policy='Run all three approved rounds; performance and domain checks are diagnostic only. Stop on execution/integrity errors or nonfinite numerical results.',
        selection_rule='Lowest one-step prediction MSE among theta0 and mature existing learners; no fixed familiarity cutoff; no familiarity-based write skipping')
    checkpoint=Path(m['checkpoint'])/'checkpoints/model_latest.pth'
    m['checkpoint_sha256']=digest(checkpoint)
    jobs=[];previous='null';counts={1:0,2:0,3:0}
    for b,j in zip(m['blocks'],old_jobs):
        target=Path(b['target']);side=json.loads(target.with_suffix('.json').read_text())
        assert side['block_sha256']==hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest()
        assert side['target_sha256']==digest(target)
        counts[b['round']]+=b['n']
        folder=out/'free_abs'/b['chunk']
        replacements={'planner.continual.familiar_thr':'0','planner.continual.familiar_read':'false',
                      'planner.continual.state_in':previous,'hydra.run.dir':str(folder)}
        argv=[f"{arg.split('=',1)[0]}={replacements[arg.split('=',1)[0]]}" if arg.split('=',1)[0] in replacements else arg for arg in j['argv']]
        with initialize_config_dir(config_dir=str(root/'conf'),version_base=None):
            cfg=compose(config_name='cont_plan_cem_rope',overrides=argv[4:])
            baseline=compose(config_name='cont_plan_cem_rope',overrides=j['argv'][4:])
        assert cfg.planner.continual.slow_lr==1e-4 and cfg.planner.adapt.steps==1
        assert cfg.planner.continual.familiar_thr==0 and not cfg.planner.continual.familiar_read
        assert cfg.planner.continual.mature_k==3 and cfg.planner.continual.abs_ratio==20
        assert cfg.planner._target_=='planning.continual_v7.CLSv7MPCPlanner'
        a=OmegaConf.to_container(cfg,resolve=True);c=OmegaConf.to_container(baseline,resolve=True)
        for key in ('familiar_thr','familiar_read','state_in'):a['planner']['continual'][key]=c['planner']['continual'][key]
        assert a==c
        if not jobs:OmegaConf.save(cfg,PLAN/'resolved_first_chunk.yaml',resolve=True)
        jobs.append(dict(chunk=b['chunk'],output=str(folder),argv=argv))
        previous=str(folder/'continual_state.pt')
    assert len(jobs)==30 and counts=={1:100,2:100,3:100} and m['blocks']==old['blocks']
    (PLAN/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (PLAN/'commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
    (PLAN/'validation.json').write_text(json.dumps(dict(status='passed_cpu_only',configs_composed=30,
        shared_target_samples=300,verified_samples_by_round=counts,only_algorithm_changes=['familiar_thr=0','familiar_read=false'],
        initial_state=None,performance_stop_removed=True),indent=2)+'\n')
    print(json.dumps(dict(gpu='1',slow_lr=1e-4,rounds=3,episodes=300,familiar_thr=0,familiar_read=False),indent=2))


if __name__=='__main__':main()
