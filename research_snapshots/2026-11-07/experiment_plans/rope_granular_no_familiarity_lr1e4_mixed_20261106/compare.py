"""Compare matched available rounds against the familiarity-on 1e-4 control."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

PLAN=Path(__file__).resolve().parent


def main():
    m=json.loads((PLAN/'manifest.json').read_text());reference=Path(m['reference_plan']);rows=[]
    for r in (1,2,3):
        own=PLAN/f'round_{r}_review.json';control=reference/f'round_{r}_review.json'
        if not own.exists() or not control.exists():continue
        a=json.loads(own.read_text());b=json.loads(control.read_text())
        for obj in ('rope','granular'):
            x=a['samples'][obj];y=b['samples'][obj]
            identity=lambda samples:[(s['chunk'],s['sample'],s['trajectory'],s['offset']) for s in samples]
            assert identity(x)==identity(y) and len(x)==50
            ca=np.asarray([s['final_cd'] for s in x]);cb=np.asarray([s['final_cd'] for s in y]);delta=ca-cb
            rng=np.random.default_rng(20261106);means=delta[rng.integers(50,size=(10000,50))].mean(axis=1)
            lo,hi=np.quantile(means,[.025,.975])
            rows.append(dict(round=r,object=obj,n=50,cd_no_familiarity_gate=float(ca.mean()),cd_original=float(cb.mean()),
                difference=float(delta.mean()),paired_bootstrap_ci95=[float(lo),float(hi)],
                wilcoxon_p=float(stats.wilcoxon(delta).pvalue) if np.any(delta) else 1.,
                domains_no_gate=a['domains'],domains_original=b['domains']))
    report=dict(comparisons=rows,note='Both use slow lr 1e-4. Negative CD difference favors removing the familiarity gate. Historical control stopped after round 1; no same-LR control exists for rounds 2/3. Single-chain exploratory comparisons.')
    (PLAN/'familiarity_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 去熟悉度门：同学习率、同目标对照','', '| 轮次 | 对象 | 去门 CD | 原方法 CD | 去门 − 原方法 |', '|---|---|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['round']} | {x['object']} | {x['cd_no_familiarity_gate']:.4f} | {x['cd_original']:.4f} | {x['difference']:+.4f} |")
    lines+=['','原1e-4对照只完成首轮；后续轮次不伪造对照、不自动新增原方法实验。','']
    (PLAN/'FAMILIARITY_COMPARISON.md').write_text('\n'.join(lines));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
