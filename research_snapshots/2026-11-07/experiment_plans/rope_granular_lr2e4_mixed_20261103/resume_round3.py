"""Complete shared round-3 targets on idle GPUs 4/7, then resume the 2e-4 chain."""
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
    with (PLAN/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute()


def execute():
    m=json.loads((PLAN/'manifest.json').read_text());out=Path(m['output']);logs=out/'pipeline_logs'
    previous=json.loads((PLAN/'pipeline_status.json').read_text())
    execution=json.loads((PLAN/'execution.json').read_text())
    assert execution['status']=='completed' and execution['completed_chunks']==20
    assert json.loads((PLAN/'round_1_review.json').read_text())['continue_authorized']
    assert (out/'free_abs/chunk_19_granular/continual_state.pt').exists()
    status={**previous,'status':'running','pid':os.getpid(),'superseded_pipeline_pid':previous['pid'],
            'resume_started_at':datetime.now(timezone.utc).isoformat(),'round':3,'target_gpus':['4','7']}
    path=PLAN/'pipeline_status.json'
    def save(**fields):
        status.update(fields,updated_at=datetime.now(timezone.utc).isoformat());tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(path)
    save(stage='preparing_round_3_on_free_devices')
    try:
        mapping=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
        needed={s.split(',')[1].strip() for s in mapping if s.split(',')[0].strip() in ('4','7')}
        assert len(needed)==2
        while True:
            active=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
            if not needed&active:break
            save(stage='waiting_for_free_devices',requested_gpus=['4','7']);time.sleep(30)
        workers=[]
        for obj,gpu in [('rope','4'),('granular','7')]:
            log=logs/f'targets_r3_{obj}_gpu{gpu}_rescheduled.log';fh=log.open('a')
            proc=subprocess.Popen([sys.executable,str(PLAN/'generate_targets.py'),obj,'--run','--round','3','--gpu',gpu],
                cwd=m['root'],env={**os.environ,'PYTHONUNBUFFERED':'1'},stdout=fh,stderr=subprocess.STDOUT)
            workers.append((proc,fh,log,gpu))
        save(stage='generating_targets',generation_workers=[dict(pid=p.pid,gpu=g,log=str(log)) for p,_,log,g in workers])
        failed=[]
        for proc,fh,log,_ in workers:
            rc=proc.wait();fh.close()
            if rc:failed.append(dict(log=str(log),returncode=rc))
        if failed:raise RuntimeError(f'Target generation failed: {failed}')
        save(stage='running_round_3',gpu='4')
        for script,args,name in [('run_chain.py',['--run','--through-round','3'],'chain_round_3.log'),
                                 ('analyze_round.py',['--round','3'],'review_round_3.log')]:
            with (logs/name).open('a') as fh:
                rc=subprocess.call([sys.executable,str(PLAN/script),*args],cwd=m['root'],
                    env={**os.environ,'PYTHONUNBUFFERED':'1'},stdout=fh,stderr=subprocess.STDOUT)
            if rc:raise RuntimeError(f'{script} failed rc={rc}; see {logs/name}')
        save(status='completed',stage='completed_3_rounds')
    except Exception as exc:
        save(status='failed',error=repr(exc));raise
    finally:save(ended_at=datetime.now(timezone.utc).isoformat())


if __name__=='__main__':main()
