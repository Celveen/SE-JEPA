"""Prepare CEM with read maturity and both familiarity protections disabled."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

PLAN=Path(__file__).resolve().parent
ROOT=PLAN.parents[1]/'CL-JEPA-cudagraph'
CONTROL=ROOT/'eval_outputs/cont_S7t3_cem/cls6a_free'
STREAM=ROOT/'data/cont_streams/S7t5'
OUT=ROOT/'eval_outputs/pushobj_no_maturity_no_familiarity_cem_lr1e4_5rounds_20261106'
REFERENCE=Path('/tmp/sejepa_no_familiarity_reference_20261106')
PYTHON='/data/zjc/envs/cljepa/bin/python'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def flat(d,prefix=''):
    out={}
    for key,value in d.items():
        name=f'{prefix}.{key}' if prefix else key
        out.update(flat(value,name) if isinstance(value,dict) else {name:value})
    return out


def main():
    if (PLAN/'manifest.json').exists() or OUT.exists():raise FileExistsError('Experiment already prepared/started')
    reference=json.loads((PLAN/'reference.json').read_text())
    core={}
    for name in ('planning/continual.py','planning/continual_v7.py','planning/continual_merge.py','planning/gd.py','planning/cem.py'):
        remote=subprocess.check_output(['git','rev-parse',f'HEAD:{name}'],cwd=REFERENCE,text=True).strip()
        local=subprocess.check_output(['git','hash-object',str(ROOT/name)],text=True).strip()
        assert local==remote,name;core[name]=local
    stream=json.loads((STREAM/'manifest.json').read_text());assert len(stream['blocks'])==35
    ckpt=Path(OmegaConf.load(CONTROL/'chunk_00_I/.hydra/config.yaml').ckpt_base_path)
    sys.path.insert(0,str(ROOT/'scripts'))
    from cont_run_stream import NAMED_ARMS
    cls='planning.continual_ablation.CLSv7NoReadMaturityMPCPlanner'
    extra=[f'planner._target_={cls}','planner.continual.slow_lr=1e-4','planner.continual.save_deltas=false',
           'planner.continual.familiar_read=false','planner.continual.familiar_thr=0',
           'planner.sub_planner.target=planning.cem.CEMPlanner','+planner.sub_planner.topk=30',
           '+planner.sub_planner.num_samples=200','planner.sub_planner.opt_steps=10']
    argv=[PYTHON,str(ROOT/'scripts/cont_run_stream.py'),'--stream','S7t5','--root',str(OUT),
          '--config','cont_plan_gd_pushobj','--arms','cls6a_free','--ckpt',str(ckpt),'--epoch','3',
          '--no-base-probes','--chain-gpus','4','--probe-gpus','4','--keep-states','6','13','20','27','34','--extra',*extra]
    hashes={};previous='null';diffs=[]
    for block in stream['blocks']:
        baseline=OmegaConf.load(CONTROL/block['chunk']/'.hydra/config.yaml')
        f=STREAM/(block['chunk']+'.pkl');hashes[f.name]=digest(f)
        assert hashes[f.name]==digest(Path(baseline.eval_data_path))
        assert (CONTROL/block['chunk']/'final_eval_results.npz').exists()
        overrides=[f'ckpt_base_path={ckpt}','model_epoch=3',f'eval_data_path={f}','n_evals=50','+wandb_logging=false',
            f"hydra.run.dir={OUT/'cls6a_free'/block['chunk']}",*NAMED_ARMS['cls6a_free'],
            f"planner.continual.tag={block['tag']}",f'planner.continual.state_in={previous}',*extra,f"seed={block['seed']}"]
        with initialize_config_dir(config_dir=str(ROOT/'conf'),version_base=None):
            cfg=compose(config_name='cont_plan_gd_pushobj',overrides=overrides)
        assert cfg.planner._target_==cls and cfg.planner.sub_planner.target=='planning.cem.CEMPlanner'
        assert cfg.planner.continual.slow_lr==1e-4 and cfg.planner.continual.mature_k==3
        assert not cfg.planner.continual.familiar_read and cfg.planner.continual.familiar_thr==0
        assert cfg.planner.continual.reject_ratio==1.2 and not cfg.planner.sub_planner.cuda_graph
        a=flat(OmegaConf.to_container(cfg,resolve=True));b=flat(OmegaConf.to_container(baseline,resolve=True))
        diff={k:[b[k],a[k]] for k in a.keys()&b.keys() if a[k]!=b[k]}
        assert set(diff)<={'planner._target_','planner.continual.familiar_thr','planner.continual.familiar_read',
                           'planner.continual.state_in','eval_data_path'},diff
        diffs.append({'chunk':block['chunk'],'differences':diff})
        if previous=='null':OmegaConf.save(cfg,PLAN/'resolved_first_chunk.yaml',resolve=True)
        previous=str(OUT/'cls6a_free'/block['chunk']/'continual_state.pt')
    train_cfg=OmegaConf.load(ckpt/'hydra.yaml')
    m=dict(status='authorized_prepared',root=str(ROOT),output=str(OUT),control=str(CONTROL),stream=str(STREAM),
        blocks=stream['blocks'],rounds=5,episodes_per_round=350,total_episodes=1750,gpu='4',slow_lr=1e-4,
        frameskip=int(train_cfg.frameskip),model_epoch=3,checkpoint=str(ckpt/'checkpoints/model_3.pth'),
        checkpoint_sha256=digest(ckpt/'checkpoints/model_3.pth'),target_sha256=hashes,planner_class=cls,
        intervention='Joint ablation: remove read maturity filter and disable familiarity read/write protection',
        familiar_read=False,familiar_thr=0,reference=reference,reference_core_blobs_identical=core,
        reference_selection='Canonical first CEM main-method stream S7t5, chosen without inspecting performance',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        implementation_sha256=digest(ROOT/'planning/continual_ablation.py'))
    env=dict(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',WANDB_MODE='offline')
    (PLAN/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (PLAN/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(ROOT),env=env),indent=2)+'\n')
    (PLAN/'validation.json').write_text(json.dumps(dict(status='passed_cpu_only',configs_composed=35,
        byte_identical_target_files=35,reference_commit=reference['commit'],core_files_identical=core,
        retained_mature_k_for_other_mechanisms=3,retained_reject_ratio=1.2,differences=diffs),indent=2)+'\n')
    print(json.dumps({k:m[k] for k in ('gpu','rounds','total_episodes','slow_lr','intervention')},indent=2))


if __name__=='__main__':main()
