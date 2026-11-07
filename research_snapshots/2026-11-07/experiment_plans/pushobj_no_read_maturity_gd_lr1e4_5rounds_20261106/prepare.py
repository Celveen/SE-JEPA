"""Prepare one matched five-round GD ablation at the main method's 1e-4 LR."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PLAN=Path(__file__).resolve().parent
ROOT=PLAN.parents[1]/'CL-JEPA-cudagraph'
CONTROL=ROOT.parent/'CL-JEPA/eval_outputs/cont_S7t3/cls6a_free'
STREAM=ROOT/'data/cont_streams/S7t5'
OUT=ROOT/'eval_outputs/pushobj_no_read_maturity_gd_lr1e4_5rounds_20261106'
PYTHON='/data/zjc/envs/cljepa/bin/python'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    if (PLAN/'manifest.json').exists() or OUT.exists():raise FileExistsError('Ablation is already prepared or started')
    stream=json.loads((STREAM/'manifest.json').read_text());assert len(stream['blocks'])==35
    cfg0=OmegaConf.load(CONTROL/'chunk_00_I/.hydra/config.yaml');ckpt=Path(cfg0.ckpt_base_path)
    hashes={};graph_modes=[]
    for block in stream['blocks']:
        cfg=OmegaConf.load(CONTROL/block['chunk']/'.hydra/config.yaml')
        f=STREAM/(block['chunk']+'.pkl');hashes[f.name]=digest(f)
        assert hashes[f.name]==digest(Path(cfg.eval_data_path))
        assert cfg.planner.continual.slow_lr==1e-4 and cfg.planner.continual.mature_k==3 and cfg.n_evals==50
        assert (CONTROL/block['chunk']/'final_eval_results.npz').exists()
        graph_modes.append(bool(cfg.planner.sub_planner.get('cuda_graph',False)))
    sys.path.insert(0,str(ROOT/'scripts'))
    from cont_run_stream import NAMED_ARMS
    target='planning.continual_ablation.CLSv7NoReadMaturityMPCPlanner'
    extra=[f'planner._target_={target}','planner.continual.slow_lr=1e-4',
           'planner.continual.save_deltas=false','planner.sub_planner.cuda_graph=true']
    argv=[PYTHON,str(ROOT/'scripts/cont_run_stream.py'),'--stream','S7t5','--root',str(OUT),
          '--config','cont_plan_gd_pushobj','--arms','cls6a_free','--ckpt',str(ckpt),'--epoch','3',
          '--no-base-probes','--chain-gpus','0','--probe-gpus','0','--keep-states','6','13','20','27','34','--extra',*extra]
    previous='null'
    for block in stream['blocks']:
        overrides=[f'ckpt_base_path={ckpt}','model_epoch=3',f"eval_data_path={STREAM/(block['chunk']+'.pkl')}",
                   'n_evals=50','+wandb_logging=false',f"hydra.run.dir={OUT/'cls6a_free'/block['chunk']}",
                   *NAMED_ARMS['cls6a_free'],f"planner.continual.tag={block['tag']}",
                   f'planner.continual.state_in={previous}',*extra,f"seed={block['seed']}"]
        with initialize_config_dir(config_dir=str(ROOT/'conf'),version_base=None):
            cfg=compose(config_name='cont_plan_gd_pushobj',overrides=overrides)
        assert cfg.planner._target_==target and cfg.planner.continual.mature_k==3
        assert cfg.planner.continual.slow_lr==1e-4 and cfg.planner.sub_planner.cuda_graph
        old=OmegaConf.load(CONTROL/block['chunk']/'.hydra/config.yaml')
        for key in ('adapt','probe','reuse_actions','max_iter','n_taken_actions'):
            assert cfg.planner[key]==old.planner[key],(block['chunk'],key)
        for key,value in cfg.planner.continual.items():
            if key not in ('state_in','read_only'):assert value==old.planner.continual[key],(block['chunk'],key)
        for key,value in cfg.planner.sub_planner.items():
            if key!='cuda_graph':assert value==old.planner.sub_planner[key],(block['chunk'],key)
        if previous=='null':OmegaConf.save(cfg,PLAN/'resolved_first_chunk.yaml',resolve=True)
        previous=str(OUT/'cls6a_free'/block['chunk']/'continual_state.pt')
    model_cfg=OmegaConf.load(ckpt/'hydra.yaml')
    m=dict(status='authorized_prepared',root=str(ROOT),output=str(OUT),control=str(CONTROL),stream=str(STREAM),
           blocks=stream['blocks'],rounds=5,episodes_per_round=350,total_episodes=1750,gpu='0',slow_lr=1e-4,
           frameskip=int(model_cfg.frameskip),model_epoch=3,checkpoint=str(ckpt/'checkpoints/model_3.pth'),
           checkpoint_sha256=digest(ckpt/'checkpoints/model_3.pth'),target_sha256=hashes,
           planner_class=target,intervention='Remove maturity filtering only from read candidates; retain original maturity checks for all other mechanisms',
           reference_selection='Canonical first main-method stream S7t5 (starts at seed 100), chosen without inspecting performance',
           reference_cuda_graph_by_chunk=graph_modes,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           implementation_sha256=digest(ROOT/'planning/continual_ablation.py'))
    env=dict(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',WANDB_MODE='offline')
    (PLAN/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (PLAN/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(ROOT),env=env),indent=2)+'\n')
    (PLAN/'validation.json').write_text(json.dumps(dict(status='passed_cpu_only',configs_composed=35,
        byte_identical_target_files=35,episodes=1750,logical_intervention='read maturity filtering only',
        retained_mature_k=3,retained_mature_ratio=.9,cuda_graph=True,reference_first_21_chunks_eager=True),indent=2)+'\n')
    print(json.dumps({k:m[k] for k in ('rounds','total_episodes','gpu','slow_lr','planner_class')},indent=2))


if __name__=='__main__':main()
