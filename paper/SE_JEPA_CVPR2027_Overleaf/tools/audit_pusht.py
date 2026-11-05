"""Recompute paired PushT aggregates; keep pending DINO-WM values out of the table."""
from pathlib import Path
import json,hashlib,re
P=Path(__file__).resolve().parents[1]
f=P/'data/revision_assets/visual_heldout_episodes.json';rows=json.loads(f.read_text())
shift=['blur','snp1','dark','red_agent','red_object','red_goal'];color=shift[-3:]
result={'source':str(f.relative_to(P)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'episode_count':len(rows),'groups':{'Shift':shift,'Color':color},'methods':{}}
for name,method in [('Frozen','frozen'),('AdaJEPA','adajepa'),('SE-JEPA','v6a')]:
 r=[x for x in rows if x['planner']=='CEM' and x['method']==method and x['domain'] in shift]
 c=[x for x in r if x['domain'] in color]
 assert r and c,(method,{x['method'] for x in rows})
 val={'shift_successes':sum(x['success'] for x in r),'shift_episodes':len(r),'shift_rate':100*sum(x['success'] for x in r)/len(r),'color_successes':sum(x['success'] for x in c),'color_episodes':len(c),'color_rate':100*sum(x['success'] for x in c)/len(c),'steps':sum(x['mpc_steps'] for x in r)/len(r)}
 result['methods'][name]=val
 prefix=r'\ours' if name=='SE-JEPA' else name
 line=next(x for x in (P/'tables/main_rows.tex').read_text().splitlines() if x.startswith(prefix))
 cells=line.split('&')[4:7]
 for cell,stat,digits in zip(cells,['shift_rate','color_rate','steps'],[1,1,2]):
  num=float(re.search(r'\d+\.\d+',cell).group());assert num==round(val[stat],digits),(name,stat,num,val[stat])
line=next(x for x in (P/'tables/main_rows.tex').read_text().splitlines() if x.startswith('DINO-WM'))
assert all(r'\missing' in x for x in line.split('&')[4:7])
s=json.loads((P/'data/paper_snapshot.json').read_text())['visual_baselines']['dinowm_VIS50']
result['DINO-WM']={'status':'withheld_from_table_pending_author_verification','snapshot_domains':s,'snapshot_shift':sum(x['rate'] for x in s.values())/6,'snapshot_color':sum(s[d]['rate'] for d in color)/3,'episode_source_available':False,'teaser':'User requested retaining the old DINO-WM visual bar at 57.0 with normal styling. Raw-episode verification remains pending; table cells stay blank.'}
(P/'data/pusht_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['methods'],indent=2));print('Paired table cells match raw-episode recomputation; DINO-WM PushT cells are withheld.')
