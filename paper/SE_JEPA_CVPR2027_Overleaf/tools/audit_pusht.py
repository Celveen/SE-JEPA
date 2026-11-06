"""Recompute PushT results for paired and external VIS50 evaluation groups."""
from pathlib import Path
import json,hashlib,re
from vis50 import external_results
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
 cells=line.split('&')[5:8]
 for cell,stat,digits in zip(cells,['shift_rate','color_rate','steps'],[1,1,2]):
  num=float(re.search(r'\d+\.\d+',cell).group());assert num==round(val[stat],digits),(name,stat,num,val[stat])
external = external_results(P/'data/vis50')
for name, val in external.items():
 line=next(x for x in (P/'tables/main_rows.tex').read_text().splitlines() if x.startswith(name))
 cells=line.split('&')[5:8]
 for cell,key,spec in zip(cells,['shift_rate','color_rate','steps'],['.1f','.1f','.2f']):
  assert cell.strip()==format(val[key],spec),(name,key,cell,val[key])
result['external_VIS50']={'source':'data/vis50/vis50_sources.json','methods':external}
(P/'data/pusht_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified all 18 Shift/Color/Steps cells against the respective episode-level sources.')
