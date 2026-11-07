"""Paired round-by-round success and MPC steps from saved results only."""
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import binomtest

PLAN=Path(__file__).resolve().parent


def finished(out):
    log=out/'logs.json'
    return (out/'final_eval_results.npz').exists() and log.exists() and 'final_eval/success_rate' in log.read_text()


def load(out,frameskip):
    cfg=OmegaConf.load(out/'.hydra/config.yaml')
    with np.load(out/'final_eval_results.npz') as d:
        success=np.asarray(d['successes'],bool)
        taken_model_steps=float(cfg.planner.n_taken_actions)/frameskip
        steps=np.where(np.isfinite(d['action_len']),np.ceil(d['action_len']/taken_model_steps),cfg.planner.max_iter)
        start=d['state_0'].copy();goal=d['state_g'].copy()
    records=[json.loads(s) for s in (out/'continual_episodes.jsonl').read_text().splitlines() if s.strip()]
    assert len(success)==len(records)==50
    assert np.array_equal(steps,np.asarray([r['steps'] for r in records]))
    return success,steps,start,goal,records


def main():
    m=json.loads((PLAN/'manifest.json').read_text());root=Path(m['output'])/'cls6a_free';control=Path(m['control'])
    rounds=[]
    for r in range(5):
        blocks=m['blocks'][r*7:(r+1)*7]
        if not all(finished(root/b['chunk']) for b in blocks):break
        sa=[];sb=[];ta=[];tb=[];reads=[]
        for b in blocks:
            a=load(root/b['chunk'],m['frameskip']);c=load(control/b['chunk'],m['frameskip'])
            assert np.array_equal(a[2],c[2]) and np.array_equal(a[3],c[3]),b['chunk']
            sa.extend(a[0]);sb.extend(c[0]);ta.extend(a[1]);tb.extend(c[1]);reads.extend(x['read'] is not None for x in a[4])
        sa=np.asarray(sa);sb=np.asarray(sb);ta=np.asarray(ta);tb=np.asarray(tb)
        gain=int((sa&~sb).sum());loss=int((~sa&sb).sum())
        rounds.append(dict(round=r+1,n=350,ablation_successes=int(sa.sum()),control_successes=int(sb.sum()),
            ablation_success_rate=float(sa.mean()),control_success_rate=float(sb.mean()),success_difference_pp=float(100*(sa.mean()-sb.mean())),
            gained_success=gain,lost_success=loss,mcnemar_p=float(binomtest(min(gain,loss),gain+loss,.5).pvalue) if gain+loss else 1.,
            ablation_mean_mpc_steps=float(ta.mean()),control_mean_mpc_steps=float(tb.mean()),
            mean_mpc_step_difference=float((ta-tb).mean()),learner_reads=sum(reads)))
    result=dict(rounds=rounds,control=m['control'],slow_lr=m['slow_lr'],intervention=m['intervention'],
        note='Matched historical CEM main-method chain. Joint removal of read maturity and familiarity read/write protections; differences cannot isolate one component. Per-round paired exploratory statistics; no multiplicity correction. Original stream retains its cross-round repeated targets.')
    (PLAN/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# PushObj：去读取成熟门及熟悉度读写保护，CEM，1e-4','',
        '| 轮次 | 完整方法 SR | 联合消融 SR | 差值 | 完整方法 MPC 步数 | 联合消融 MPC 步数 |',
        '|---|---:|---:|---:|---:|---:|']
    for x in rounds:
        lines.append(f"| {x['round']} | {x['control_success_rate']*100:.2f}% | {x['ablation_success_rate']*100:.2f}% | {x['success_difference_pp']:+.2f} pp | {x['control_mean_mpc_steps']:.2f} | {x['ablation_mean_mpc_steps']:.2f} |")
    lines += ['', '同一目标、次序、种子、基座和1e-4慢学习率。MPC步数从已保存动作长度和配置离线恢复，没有额外仿真。', '']
    (PLAN/'COMPARISON.md').write_text('\n'.join(lines));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
