"""Launch the single authorized ablation chain and update completed-round reports."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

PLAN=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    job=json.loads((PLAN/'command.json').read_text());print(shlex.join(job['argv']),flush=True)
    if not a.run:return
    with (PLAN/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(job)


def execute(job):
    m=json.loads((PLAN/'manifest.json').read_text());root=Path(m['output'])/'cls6a_free'
    assert len(m['blocks'])==35 and m['total_episodes']==1750 and m['slow_lr']==1e-4
    for name,sha in m['target_sha256'].items():assert hashlib.sha256((Path(m['stream'])/name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((Path(m['root'])/'planning/continual_ablation.py').read_bytes()).hexdigest()==m['implementation_sha256']
    status=dict(status='running',pid=os.getpid(),gpu=m['gpu'],slow_lr=m['slow_lr'],started_at=datetime.now(timezone.utc).isoformat())
    path=PLAN/'execution.json'
    def save(**fields):
        status.update(fields,updated_at=datetime.now(timezone.utc).isoformat());tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(path)
    done=set();reported=0;save()
    proc=subprocess.Popen(job['argv'],cwd=job['cwd'],env={**os.environ,**job['env']})
    save(runner_pid=proc.pid)
    try:
        while True:
            try:rc=proc.wait(timeout=30)
            except subprocess.TimeoutExpired:rc=None
            for b in m['blocks']:
                if b['chunk'] in done:continue
                out=root/b['chunk'];log=out/'logs.json'
                if (out/'final_eval_results.npz').exists() and log.exists() and 'final_eval/success_rate' in log.read_text():done.add(b['chunk'])
            save(completed_chunks=len(done),completed_episodes=len(done)*50)
            completed_rounds=0
            for r in range(5):
                if all(b['chunk'] in done for b in m['blocks'][r*7:(r+1)*7]):completed_rounds=r+1
                else:break
            if completed_rounds>reported:
                with (PLAN/'analysis.log').open('a') as f:
                    result=subprocess.run([sys.executable,str(PLAN/'analyze.py')],cwd=job['cwd'],
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},stdout=f,stderr=subprocess.STDOUT)
                save(completed_rounds=completed_rounds,analysis_returncode=result.returncode)
                if result.returncode==0:reported=completed_rounds
            if rc is not None:break
        ok=rc==0 and len(done)==35
        save(status=('completed' if reported==5 else 'completed_needs_analysis') if ok else 'failed',returncode=rc,expected_chunks=35)
        if not ok:raise RuntimeError(f'Incomplete ablation: rc={rc}, {len(done)}/35 chunks')
    except Exception as exc:
        save(status='failed',error=repr(exc));raise
    finally:save(ended_at=datetime.now(timezone.utc).isoformat())


if __name__=='__main__':main()
