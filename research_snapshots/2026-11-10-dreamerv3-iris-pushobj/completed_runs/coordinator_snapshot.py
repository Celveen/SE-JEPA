"""Approved eight-GPU frozen evaluation. Default prints jobs without running."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run import PLAN,PYTHON,TAGS,summarize


def now():return datetime.now(timezone.utc).isoformat()


def save_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def jobs_for(manifest):
    jobs=[dict(id='dreamerv3_all',method='dreamerv3',gpu='0',domains=TAGS,worker='all')]
    jobs.extend(dict(id=f'iris_{tag}',method='iris',gpu=str(index+1),domains=[tag],worker=tag) for index,tag in enumerate(TAGS))
    for job in jobs:
        job['command']=[PYTHON,'-u',str(PLAN/'run.py'),job['method'],'--gpu',job['gpu'],'--domains',*job['domains'],'--run']
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    manifest=json.loads((PLAN/'manifest.json').read_text());jobs=jobs_for(manifest)
    if not args.run:
        print(json.dumps(dict(dry_run=True,jobs=jobs,total_episodes=700,training=False),indent=2));return
    assert manifest['authorization']['experiment_launch']=='confirmed by user: 启动吧; eight-GPU scheme'
    def interrupted(signum,frame):raise InterruptedError(f'Coordinator received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    with (PLAN/'.coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).strip()
        if active:raise RuntimeError(f'GPU processes appeared before startup; inspect: {active}')
        output=Path(manifest['output_root']);output.mkdir(parents=True,exist_ok=False)
        shutil.copy2(PLAN/'manifest.json',output/'input_manifest.json')
        shutil.copy2(PLAN/'run_all.py',output/'coordinator_snapshot.py')
        state=dict(status='starting',pid=os.getpid(),started_at=now(),planned_episodes=700,
            jobs={j['id']:dict(status='queued',method=j['method'],gpu=j['gpu'],domains=j['domains']) for j in jobs})
        processes={};handles=[];summarized=set()
        def persist():
            state['updated_at']=now()
            save_json(PLAN/'pipeline_status.json',state);save_json(output/'pipeline_status.json',state)
        persist()
        try:
            for job in jobs:
                handle=(output/f"{job['id']}_launcher.log").open('w');handles.append(handle)
                process=subprocess.Popen(job['command'],cwd=manifest['root'],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
                processes[job['id']]=process;state['jobs'][job['id']].update(status='starting',pid=process.pid,started_at=now());persist()
            state['status']='evaluating'
            while True:
                for job in jobs:
                    item=state['jobs'][job['id']];process=processes[job['id']]
                    worker_file=output/job['method']/'_workers'/job['worker']/'pipeline_status.json'
                    if worker_file.exists():
                        item['worker']=json.loads(worker_file.read_text())
                        if process.poll() is None:item['status']=item['worker']['status']
                    if process.poll() is not None:item.update(status='completed' if process.returncode==0 else 'failed',exit_code=process.returncode)
                    item['episodes_saved']=sum(len(list((output/job['method']/tag/'episodes').glob('episode_*.npz'))) for tag in job['domains'])
                for method in ['dreamerv3','iris']:
                    method_jobs=[state['jobs'][j['id']] for j in jobs if j['method']==method]
                    if method not in summarized and all(j['status']=='completed' for j in method_jobs):
                        summary=summarize(manifest,method,TAGS)
                        save_json(output/method/'summary.json',summary)
                        save_json(output/method/'pipeline_status.json',dict(status='completed',ended_at=now(),summary=summary));summarized.add(method)
                state['episodes_saved']=sum(j['episodes_saved'] for j in state['jobs'].values())
                all_ended=all(p.poll() is not None for p in processes.values())
                if all_ended:
                    state.update(status='completed' if len(summarized)==2 else 'failed',ended_at=now());persist()
                    if state['status']=='failed':raise RuntimeError('Worker failed; outputs retained for inspection')
                    break
                persist();time.sleep(10)
        except BaseException as exc:
            state.update(status='failed',error=repr(exc),ended_at=now());persist();raise
        finally:
            for process in processes.values():
                if process.poll() is None:process.terminate()
            for process in processes.values():
                if process.poll() is None:
                    try:process.wait(timeout=25)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
            for handle in handles:handle.close()


if __name__=='__main__':main()
