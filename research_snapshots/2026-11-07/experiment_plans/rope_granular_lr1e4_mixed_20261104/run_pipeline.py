"""Run the matched 1e-4 stream; review round 1 before continuing."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PLAN=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if not args.run:
        print('Dry run: CEM mixed stream, 1e-4, GPU 1, 3 rounds; reuse target files; round-1 review gate.')
        return
    with (PLAN/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute()


def execute():
    m=json.loads((PLAN/'manifest.json').read_text());out=Path(m['output']);logs=out/'pipeline_logs';logs.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',pid=os.getpid(),gpu=m['chain_gpu'],slow_lr=m['slow_lr'],started_at=datetime.now(timezone.utc).isoformat())
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
            needed=[Path(b['target']) for b in m['blocks'] if b['round']==r]
            while not all(t.exists() and t.with_suffix('.json').exists() for t in needed):
                source_status=Path(m['target_source_plan'])/'pipeline_status.json'
                if source_status.exists() and json.loads(source_status.read_text()).get('status')=='failed':
                    raise RuntimeError(f'Shared target preparation failed; see {source_status}')
                save(stage='waiting_for_shared_targets',round=r);time.sleep(30)
            save(stage=f'running_round_{r}',round=r)
            call('run_chain.py',['--run','--through-round',str(r)],f'chain_round_{r}.log')
            save(stage=f'reviewing_round_{r}')
            call('analyze_round.py',['--round',str(r)],f'review_round_{r}.log')
            call('compare.py',[],f'compare_round_{r}.log')
            review=json.loads((PLAN/f'round_{r}_review.json').read_text())
            save(completed_rounds=r,last_performance=review['performance'],last_domains=review['domains'])
            if r==1 and not review['continue_authorized']:
                save(status='needs_review',stage='stopped_after_round_1',checks=review['checks']);return
        save(status='completed',stage='completed_3_rounds')
    except Exception as exc:
        save(status='failed',error=repr(exc));raise
    finally:save(ended_at=datetime.now(timezone.utc).isoformat())


if __name__=='__main__':main()
