"""Continue a historical 1e-4 slow-only chain for rounds 4 and 5 only."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PLAN=Path(__file__).resolve().parent


def finished(out):
    logs=out/'logs.json'; rows=out/'continual_episodes.jsonl'
    return ((out/'final_eval_results.npz').exists() and logs.exists()
            and 'final_eval/mean_chamfer_distance' in logs.read_text()
            and rows.exists() and len(rows.read_text().splitlines())==10)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('object',choices=['rope','granular'])
    ap.add_argument('--run',action='store_true'); a=ap.parse_args()
    m=json.loads((PLAN/'manifest.json').read_text()); obj=m['objects'][a.object]
    if not a.run:
        print(f"Dry run: {a.object}, GPU {obj['gpu']}, resume 150 episodes and add rounds 4/5 only; lr=1e-4, fast steps=0")
        return
    with (PLAN/f'{a.object}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(m,obj)


def execute(m,obj):
    assert len(obj['jobs'])==10 and {j['round'] for j in obj['jobs']}=={4,5}
    initial_hash=hashlib.sha256()
    with Path(obj['initial_state']).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): initial_hash.update(part)
    assert initial_hash.hexdigest()==obj['initial_state_sha256'], 'Historical round-3 state changed'
    for target in obj['targets']:
        assert hashlib.sha256(Path(target['path']).read_bytes()).hexdigest()==target['sha256'],target['path']
    root=Path(m['root']); flex=root.parent/'PyFleX'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':obj['gpu'],'EGL_GPU':obj['gpu'],'PYFLEXROOT':str(flex),
         'DATASET_DIR':str(root.parent/'CL-JEPA/data'),'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1',
         'WANDB_MODE':'offline','SDL_VIDEODRIVER':'dummy','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
         'OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    env['PYTHONPATH']=str(flex/'bindings/build')+':'+env.get('PYTHONPATH','')
    env['LD_LIBRARY_PATH']=str(flex/'external/SDL2-2.0.4/lib/x64')+':'+env.get('LD_LIBRARY_PATH','')
    status=dict(status='running',object=obj['object'],gpu=obj['gpu'],pid=os.getpid(),
                initial_state=obj['initial_state'],initial_episodes=150,slow_lr=1e-4,
                started_at=datetime.now(timezone.utc).isoformat())
    status_path=PLAN/f"execution_{obj['object']}.json"
    def save(**fields):
        status.update(fields,updated_at=datetime.now(timezone.utc).isoformat())
        tmp=status_path.with_suffix('.tmp'); tmp.write_text(json.dumps(status,indent=2)+'\n'); tmp.replace(status_path)
    jobs=obj['jobs']; save()
    try:
        start=0
        for i,job in enumerate(jobs):
            out=Path(job['output'])
            if not finished(out): break
            if (out/'continual_state.pt').exists(): start=i+1
        for i in range(start,len(jobs)):
            job=jobs[i]; out=Path(job['output']); out.mkdir(parents=True,exist_ok=True)
            save(round=job['round'],chunk=job['chunk'],current_output=str(out))
            print(f"START {obj['object']} round={job['round']} chunk={job['chunk']} GPU={obj['gpu']}",flush=True)
            with (out/'console.log').open('w') as f:
                rc=subprocess.call(job['argv'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            if rc or not finished(out) or not (out/'continual_state.pt').exists():
                raise RuntimeError(f"{out}: incomplete/failed, rc={rc}")
            save(completed_new_chunks=i+1,completed_new_episodes=(i+1)*10,total_episodes=150+(i+1)*10)
            print(f"DONE {obj['object']} round={job['round']} chunk={job['chunk']}",flush=True)
            if i>0 and i-1 not in obj['keep_state_indices']:
                previous=Path(jobs[i-1]['output'])/'continual_state.pt'
                if previous.exists(): previous.unlink()
            if (i+1)%5==0:
                log=Path(obj['output'])/f"analysis_round_{job['round']}.log"
                with log.open('w') as f:
                    rc=subprocess.call([sys.executable,str(PLAN/'analyze.py'),obj['object']],cwd=root,
                                       env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2',
                                            'OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/cljepa_single_rounds_mpl'},
                                       stdout=f,stderr=subprocess.STDOUT)
                if rc: raise RuntimeError(f"Round analysis failed; see {log}")
                save(completed_rounds=job['round'],summary=str(Path(obj['output'])/'analysis/SUMMARY.md'))
        save(status='completed',completed_rounds=5)
    except Exception as exc:
        save(status='failed',error=repr(exc)); raise
    finally:
        save(ended_at=datetime.now(timezone.utc).isoformat())


if __name__=='__main__': main()
