"""Prepare the same joint ablation as CEM, using accelerated GD on GPU 5."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

PLAN=Path(__file__).resolve().parent
CEM=PLAN.parent/'pushobj_no_maturity_no_familiarity_cem_lr1e4_5rounds_20261106'
GD=PLAN.parent/'pushobj_no_read_maturity_gd_lr1e4_5rounds_20261106'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def flat(d,prefix=''):
    out={}
    for k,v in d.items():
        key=f'{prefix}.{k}' if prefix else k
        out.update(flat(v,key) if isinstance(v,dict) else {key:v})
    return out


def main():
    if (PLAN/'manifest.json').exists():raise FileExistsError('Already prepared')
    cem=json.loads((CEM/'manifest.json').read_text());gd=json.loads((GD/'manifest.json').read_text())
    assert cem['blocks']==gd['blocks'] and cem['target_sha256']==gd['target_sha256']
    root=Path(cem['root']);out=root/'eval_outputs/pushobj_no_maturity_no_familiarity_gd_lr1e4_5rounds_20261106'
    if out.exists():raise FileExistsError(out)
    assert digest(root/'planning/continual_ablation.py')==cem['implementation_sha256']
    assert digest(cem['checkpoint'])==cem['checkpoint_sha256']
    for name,blob in cem['reference_core_blobs_identical'].items():
        assert subprocess.check_output(['git','hash-object',str(root/name)],text=True).strip()==blob,name
    command=copy.deepcopy(json.loads((GD/'command.json').read_text()));argv=command['argv']
    for flag,value in [('--root',str(out)),('--chain-gpus','5'),('--probe-gpus','5')]:argv[argv.index(flag)+1]=value
    argv.extend(['planner.continual.familiar_read=false','planner.continual.familiar_thr=0'])
    extras=argv[argv.index('--extra')+1:]
    sys.path.insert(0,str(root/'scripts'))
    from cont_run_stream import NAMED_ARMS
    control=Path(gd['control']);stream=Path(cem['stream']);previous='null';differences=[]
    for block in cem['blocks']:
        f=stream/(block['chunk']+'.pkl');assert digest(f)==cem['target_sha256'][f.name]
        old=OmegaConf.load(control/block['chunk']/'.hydra/config.yaml')
        assert digest(Path(old.eval_data_path))==digest(f)
        overrides=[f'ckpt_base_path={old.ckpt_base_path}','model_epoch=3',f'eval_data_path={f}',
            'n_evals=50','+wandb_logging=false',f"hydra.run.dir={out/'cls6a_free'/block['chunk']}",
            *NAMED_ARMS['cls6a_free'],f"planner.continual.tag={block['tag']}",
            f'planner.continual.state_in={previous}',*extras,f"seed={block['seed']}"]
        with initialize_config_dir(config_dir=str(root/'conf'),version_base=None):
            cfg=compose(config_name='cont_plan_gd_pushobj',overrides=overrides)
        assert cfg.planner._target_==cem['planner_class']
        assert cfg.planner.continual.slow_lr==1e-4 and cfg.planner.continual.mature_k==3
        assert cfg.planner.continual.familiar_thr==0 and not cfg.planner.continual.familiar_read
        assert cfg.planner.continual.reject_ratio==1.2
        assert cfg.planner.sub_planner.target=='planning.gd.GDPlanner' and cfg.planner.sub_planner.cuda_graph
        assert cfg.planner.sub_planner.lr==.1 and cfg.planner.sub_planner.opt_steps==100
        a=flat(OmegaConf.to_container(cfg,resolve=True));b=flat(OmegaConf.to_container(old,resolve=True))
        diff={k:[b[k],a[k]] for k in a.keys()&b.keys() if a[k]!=b[k]}
        assert set(diff)<={'planner._target_','planner.continual.familiar_thr','planner.continual.familiar_read',
                           'planner.continual.state_in','planner.sub_planner.cuda_graph','eval_data_path'},diff
        differences.append({'chunk':block['chunk'],'differences':diff})
        if previous=='null':OmegaConf.save(cfg,PLAN/'resolved_first_chunk.yaml',resolve=True)
        previous=str(out/'cls6a_free'/block['chunk']/'continual_state.pt')
    m=copy.deepcopy(cem)
    m.update(status='authorized_prepared',output=str(out),control=str(control),gpu='5',planner='GD',cuda_graph=True,
             matching_joint_cem_plan=str(CEM),matching_read_maturity_only_gd_plan=str(GD),
             reference_selection='Same S7t5 targets as joint CEM and canonical first GD main-method stream',
             commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    (PLAN/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (PLAN/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    (PLAN/'validation.json').write_text(json.dumps(dict(status='passed_cpu_only',configs_composed=35,
        joint_rule_matches_cem=True,targets_match_cem_and_gd_control=True,implementation_hash_unchanged=True,
        behavior_tests_reused_from=str(CEM/'behavior_validation.json'),differences=differences),indent=2)+'\n')
    print(json.dumps({k:m[k] for k in ('gpu','planner','cuda_graph','slow_lr','rounds','total_episodes','intervention')},indent=2))


if __name__=='__main__':main()
