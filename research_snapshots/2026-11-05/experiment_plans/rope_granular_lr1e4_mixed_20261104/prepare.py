"""Prepare a matched 1e-4 mixed stream using the existing 2e-4 stream's targets."""
import copy
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PLAN=Path(__file__).resolve().parent
SOURCE=PLAN.parent/'rope_granular_lr2e4_mixed_20261103'


def main():
    if (PLAN/'manifest.json').exists(): raise FileExistsError('Manifest already prepared')
    source=json.loads((SOURCE/'manifest.json').read_text())
    source_jobs=json.loads((SOURCE/'commands.json').read_text())
    root=Path(source['root']);output=root/'eval_outputs/rope_granular_lr1e4_mixed_20261104'
    m=copy.deepcopy(source)
    for key in ('generation_shards','generator_gpus'):m.pop(key,None)
    m.update(status='authorized_prepared',slow_lr=1e-4,chain_gpu='1',output=str(output),
             target_source_plan=str(SOURCE),first_round_policy='Same conservative round-1 checks as the matched 2e-4 stream')
    jobs=[];previous='null';available={1:0,2:0,3:0}
    for block,old in zip(m['blocks'],source_jobs):
        target=Path(block['target']);sidecar=target.with_suffix('.json')
        if target.exists() and sidecar.exists():
            record=json.loads(sidecar.read_text())
            assert record['block_sha256']==hashlib.sha256(json.dumps(block,sort_keys=True).encode()).hexdigest()
            assert record['target_sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
            available[block['round']]+=block['n']
        out=output/'free_abs'/block['chunk']
        if out.exists():raise FileExistsError(out)
        replacements={'planner.continual.slow_lr':'1e-4','planner.continual.state_in':previous,'hydra.run.dir':str(out)}
        argv=[f"{arg.split('=',1)[0]}={replacements[arg.split('=',1)[0]]}" if arg.split('=',1)[0] in replacements else arg for arg in old['argv']]
        with initialize_config_dir(config_dir=str(root/'conf'),version_base=None):
            new_cfg=compose(config_name='cont_plan_cem_rope',overrides=argv[4:])
            old_cfg=compose(config_name='cont_plan_cem_rope',overrides=old['argv'][4:])
        assert new_cfg.planner.continual.slow_lr==1e-4 and new_cfg.planner.adapt.steps==1
        assert new_cfg.planner.continual.write=='delta' and new_cfg.planner.continual.abs_ratio==20
        # Aside from the learning rate and separate state/output paths, all planner settings match.
        a=OmegaConf.to_container(new_cfg,resolve=True);b=OmegaConf.to_container(old_cfg,resolve=True)
        a['planner']['continual']['slow_lr']=b['planner']['continual']['slow_lr']
        a['planner']['continual']['state_in']=b['planner']['continual']['state_in']
        assert a==b
        if len(jobs)<2:OmegaConf.save(new_cfg,PLAN/f"resolved_first_{block['object']}.yaml",resolve=True)
        jobs.append(dict(chunk=block['chunk'],output=str(out),argv=argv))
        previous=str(out/'continual_state.pt')
    assert m['blocks']==source['blocks'] and available[1]==available[2]==100
    assert 'planner.continual.state_in=null' in jobs[0]['argv']
    (PLAN/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (PLAN/'commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
    (PLAN/'validation.json').write_text(json.dumps(dict(status='passed_cpu_only',jobs=30,episodes=300,
        shared_target_blocks_identical=True,only_hyperparameter_change='slow_lr: 2e-4 -> 1e-4',
        verified_target_counts=available,initial_state=None,chain_gpu='1'),indent=2)+'\n')
    print(json.dumps({'status':'prepared','target_counts':available,'chain_gpu':'1','slow_lr':1e-4},indent=2))


if __name__=='__main__':main()
