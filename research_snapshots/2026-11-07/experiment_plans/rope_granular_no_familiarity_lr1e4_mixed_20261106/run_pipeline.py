"""Run all three rounds; keep domain/performance reporting separate from integrity checks."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

PLAN=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:
        print('Dry run: GPU 1, CEM, slow lr 1e-4, familiarity threshold disabled, all 3 rounds / 300 episodes.')
        return
    with (PLAN/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute()


def execute():
    m=json.loads((PLAN/'manifest.json').read_text());logs=Path(m['output'])/'pipeline_logs';logs.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',pid=os.getpid(),gpu=m['chain_gpu'],slow_lr=m['slow_lr'],
                started_at=datetime.now(timezone.utc).isoformat(),continuation_policy=m['continuation_policy'])
    path=PLAN/'pipeline_status.json'
    def save(**fields):
        status.update(fields,updated_at=datetime.now(timezone.utc).isoformat());tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(path)
    def call(script,args,name):
        with (logs/name).open('a') as f:
            rc=subprocess.call([sys.executable,str(PLAN/script),*args],cwd=m['root'],
                env={**os.environ,'PYTHONUNBUFFERED':'1'},stdout=f,stderr=subprocess.STDOUT)
        if rc:raise RuntimeError(f'{script} failed rc={rc}; see {logs/name}')
    save(stage='starting')
    try:
        for r in (1,2,3):
            save(stage=f'running_round_{r}',round=r)
            call('run_chain.py',['--run','--through-round',str(r)],f'chain_round_{r}.log')
            save(stage=f'analyzing_round_{r}')
            call('analyze_round.py',['--round',str(r)],f'analysis_round_{r}.log')
            report=json.loads((PLAN/f'round_{r}_review.json').read_text())
            save(completed_rounds=r,last_performance=report['performance'],last_domains=report['domains'])
            if not report['integrity_ok']:
                raise RuntimeError(f'Round {r} has incomplete/nonfinite results; see saved report')
            call('compare.py',[],f'compare_round_{r}.log')
        save(status='completed',stage='completed_3_rounds')
    except Exception as exc:
        save(status='failed',error=repr(exc));raise
    finally:save(ended_at=datetime.now(timezone.utc).isoformat())


if __name__=='__main__':main()
