"""Compare matched rounds of mixed 1e-4 and 2e-4 runs using saved CD values."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

PLAN=Path(__file__).resolve().parent


def main():
    m=json.loads((PLAN/'manifest.json').read_text());source=Path(m['target_source_plan'])
    rows=[]
    for r in (1,2,3):
        low=PLAN/f'round_{r}_review.json';high=source/f'round_{r}_review.json'
        if not low.exists() or not high.exists():continue
        a=json.loads(low.read_text());b=json.loads(high.read_text())
        for obj in ('rope','granular'):
            x=a['samples'][obj];y=b['samples'][obj]
            keys=lambda samples:[(s['chunk'],s['sample'],s['trajectory'],s['offset']) for s in samples]
            assert keys(x)==keys(y) and len(x)==50
            ca=np.asarray([s['final_cd'] for s in x]);cb=np.asarray([s['final_cd'] for s in y])
            delta=ca-cb;rng=np.random.default_rng(20261104)
            means=delta[rng.integers(50,size=(10000,50))].mean(axis=1)
            lo,hi=np.quantile(means,[.025,.975])
            rows.append(dict(round=r,object=obj,n=50,cd_1e4=float(ca.mean()),cd_2e4=float(cb.mean()),
                cd_1e4_minus_2e4=float(delta.mean()),paired_bootstrap_ci95=[float(lo),float(hi)],
                wilcoxon_p=float(stats.wilcoxon(delta).pvalue) if np.any(delta) else 1.,
                domains_1e4=a['domains'],domains_2e4=b['domains'],purity_1e4=a['member_weighted_purity'],purity_2e4=b['member_weighted_purity']))
    report=dict(comparisons=rows,note='Same targets, order, seeds and mixed method. Negative CD difference favors 1e-4. Single-chain exploratory comparison; no multiple-testing adjustment.')
    (PLAN/'lr_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 混合域：同目标同轮次的学习率对照','', '| 轮次 | 对象 | 1e-4 CD | 2e-4 CD | 1e-4 − 2e-4 | 配对 95% 区间 |',
           '|---|---|---:|---:|---:|---|']
    for row in rows:
        lo,hi=row['paired_bootstrap_ci95']
        lines.append(f"| {row['round']} | {row['object']} | {row['cd_1e4']:.4f} | {row['cd_2e4']:.4f} | {row['cd_1e4_minus_2e4']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines+=['','差值为负表示 1e-4 的 CD 更低；领域组成和纯度见 JSON。','']
    (PLAN/'LR_COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
