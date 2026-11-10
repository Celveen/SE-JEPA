"""Reviewable evaluation-only launcher. Default prints the plan without running."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

PLAN=Path(__file__).resolve().parent
PYTHON='/data/zjc/envs/cljepa/bin/python'
DREAMER_PYTHON='/data/zjc/adaJEPA/baselines/dreamerv3-env/bin/python'
TAGS=['I','L','square','Z','small_tee','plus','T']


def summarize(manifest,method,domains):
    import numpy as np
    output=Path(manifest['output_root'])/method
    results={}
    for block in manifest['blocks']:
        if block['tag'] not in domains:continue
        with np.load(output/block['tag']/'final_eval_results.npz') as data,np.load(block['target']) as target:
            assert len(data['successes'])==50
            for name in ['state_0','state_g']:np.testing.assert_array_equal(data[name],target[name])
            steps=data['executed_mpc_steps'];assert np.all((steps>=1)&(steps<=30))
            np.testing.assert_array_equal(data['physical_action_count'],5*steps)
            np.testing.assert_array_equal(steps,np.where(data['successes'],data['action_len'],30))
            assert np.isfinite(data['actions']).all() and np.isfinite(data['final_states']).all()
            results[block['shape']]=dict(n=50,successes=int(data['successes'].sum()),success_rate=float(data['successes'].mean()),
                mean_mpc_steps=float(steps.mean()),mean_physical_actions=float(data['physical_action_count'].mean()))
        frozen=json.loads((output/block['tag']/'frozen_weights_check.json').read_text())
        assert frozen['unchanged'] and frozen['before_sha256']==frozen['after_sha256']
    assert len(results)==len(domains)
    summary=dict(method=method,episodes=50*len(domains),seed=2000,training_shapes=['T'],
        protocol='T-only offline world model with CEM transferred to PushObj',by_shape=results,
        success_rate=sum(item['successes'] for item in results.values())/(50*len(domains)))
    if len(domains)==7:
        summary.update(all_success_rate=summary['success_rate'],T_seen_success_rate=results['T']['success_rate'],
            six_unseen_success_rate=sum(v['successes'] for k,v in results.items() if k!='T')/300,
            I_square_small_tee_success_rate=sum(results[k]['successes'] for k in ['I','square','small_tee'])/150)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('method',choices=['dreamerv3','iris'])
    ap.add_argument('--gpu',required=True);ap.add_argument('--domains',nargs='+',choices=TAGS,default=TAGS)
    ap.add_argument('--run',action='store_true');a=ap.parse_args()
    assert len(a.domains)==len(set(a.domains))
    m=json.loads((PLAN/'manifest.json').read_text());model=m['models'][a.method]
    output=Path(m['output_root'])/a.method;socket=Path(f'/tmp/sejepa_shape_{a.method}_{os.getpid()}.sock')
    worker='all' if a.domains==TAGS else '_'.join(a.domains)
    worker_output=output/'_workers'/worker
    interpreter=DREAMER_PYTHON if a.method=='dreamerv3' else PYTHON
    server_cmd=[interpreter,'-u',str(PLAN/'code/serve_frozen.py'),a.method,'--checkpoint-dir',model['checkpoint'],'--socket',str(socket),'--run']
    hashes=model['checkpoint_sha256'];checksum=hashes['model_latest.pkl'] if a.method=='dreamerv3' else hashlib.sha256((hashes['world_model.pt']+hashes['tokenizer_model.pt']).encode()).hexdigest()
    eval_cmd=[PYTHON,'-u',str(PLAN/'code/evaluate.py'),'--method',a.method,'--checkpoint-dir',model['checkpoint'],
        '--checkpoint-sha256',checksum,'--socket',str(socket),'--output',str(output),'--target-root',str(PLAN/'targets'),
        '--seed','2000','--episodes','50','--samples','200','--topk','30','--iterations','10','--max-replans','30','--domains',*a.domains,'--run']
    if not a.run:
        print(json.dumps(dict(dry_run=True,gpu=a.gpu,method=a.method,domains=a.domains,training=False,server_command=server_cmd,evaluation_command=eval_cmd),indent=2));return
    def interrupted(signum,frame):raise InterruptedError(f'Worker received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    with (PLAN/f'.{a.method}_{worker}.lock').open('a') as lock,(Path(m['root'])/f'eval_outputs/baselines/.offline_visual_gpu_{a.gpu}.lock').open('a') as gpu_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        mapping=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
        uid=next(s.split(',')[1].strip() for s in mapping if s.split(',')[0].strip()==a.gpu)
        active=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
        if uid in active:raise RuntimeError('GPU is occupied; inspect before launch')
        for name,sha in hashes.items():assert hashlib.sha256((Path(model['checkpoint'])/name).read_bytes()).hexdigest()==sha
        for b in m['blocks']:assert hashlib.sha256(Path(b['target']).read_bytes()).hexdigest()==b['target_sha256']
        for name,sha in m['prepared_code_sha256'].items():assert hashlib.sha256((PLAN/name).read_bytes()).hexdigest()==sha
        for name,sha in m['runtime_dependency_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        assert not any((output/tag).exists() for tag in a.domains),'Existing results; inspect before resuming'
        worker_output.mkdir(parents=True,exist_ok=False)
        shutil.copy2(PLAN/'manifest.json',worker_output/'input_manifest.json')
        shutil.copytree(PLAN/'code',worker_output/'code_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(PLAN/'run.py',worker_output/'run_snapshot.py')
        (worker_output/'launch_commands.json').write_text(json.dumps(dict(server=server_cmd,evaluator=eval_cmd,gpu=a.gpu,domains=a.domains),indent=2)+'\n')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':uid,'XLA_PYTHON_CLIENT_PREALLOCATE':'false',
             'JAX_COMPILATION_CACHE_DIR':'/data/zjc/adaJEPA/baselines/offline_visual_cache/jax_compilation',
             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','SDL_VIDEODRIVER':'dummy','PYTHONUNBUFFERED':'1'}
        status=dict(status='starting',method=a.method,gpu=a.gpu,gpu_uuid=uid,domains=a.domains,pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat())
        def save(**updates):
            status.update(updates,updated_at=datetime.now(timezone.utc).isoformat())
            temporary=worker_output/'pipeline_status.json.tmp'
            temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(worker_output/'pipeline_status.json')
        save();server=evaluator=None
        try:
            with (worker_output/'server.log').open('w') as log:
                server=subprocess.Popen(server_cmd,cwd=m['root'],env=env,stdout=log,stderr=subprocess.STDOUT)
                save(server_pid=server.pid)
                started=time.monotonic()
                while not socket.exists():
                    if server.poll() is not None:raise RuntimeError('Inference server failed')
                    if time.monotonic()-started>300:raise TimeoutError('Inference server startup')
                    time.sleep(1)
                with (worker_output/'console.log').open('w') as f:
                    evaluator=subprocess.Popen(eval_cmd,cwd=m['root'],env=env,stdout=f,stderr=subprocess.STDOUT)
                    save(status='evaluating',evaluator_pid=evaluator.pid);rc=evaluator.wait()
                if rc:raise RuntimeError(f'Evaluation failed rc={rc}')
            summary=summarize(m,a.method,a.domains)
            (worker_output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');save(status='completed',summary=summary)
        except BaseException as exc:save(status='failed',error=repr(exc));raise
        finally:
            for process in [evaluator,server]:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            if socket.exists():socket.unlink()
            save(ended_at=datetime.now(timezone.utc).isoformat())


if __name__=='__main__':main()
