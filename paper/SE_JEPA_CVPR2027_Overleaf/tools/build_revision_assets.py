"""Rebuild revised figures from archived Drive frames and numeric summaries.
Run after build_paper_assets.py; this overrides its screenshot-based qualitative panel.
"""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from restyle_paper_figures import METHOD_COLORS, METHOD_STYLES, TEXT, set_style
from PIL import Image
P=Path(__file__).resolve().parents[1]; D=P/'data/revision_assets'; F=P/'figures'
set_style()
methods=['frozen','adajepa','ours']; names={'frozen':'Frozen','adajepa':'AdaJEPA','ours':'SE-JEPA'}; colors=METHOD_COLORS
cases={k:json.loads((D/(k+'_case.json')).read_text()) for k in ['shape','visual']}
def frame(ax,k,m,step):
 f=D/k/m/(('goal' if step=='goal' else f'step_{step:02d}')+'.png')
 ax.imshow(Image.open(f),interpolation='none');ax.set_xticks([]);ax.set_yticks([])
 for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#dce1e5');sp.set_linewidth(.55)
def save(fig,n):
 fig.savefig(F/(n+'.pdf'),bbox_inches='tight',pad_inches=.025)
 fig.savefig(F/(n+'.png'),bbox_inches='tight',pad_inches=.025,dpi=300)
 plt.close(fig)
# Selected failures/successes share pixel-identical initial and goal observations.
# Equal-size crops enlarge the raw simulator frames without changing their pixels.
selection=json.loads((D/'selection_manifest.json').read_text())['cases']
fig, axes = plt.subplots(len(selection), 5, figsize=(7, 2.55))
fig.subplots_adjust(left=.085, right=.995, top=.87, bottom=.15, wspace=.04, hspace=.33)
for row, case in enumerate(selection):
 assert not case['methods']['frozen']['success']
 assert not case['methods']['adajepa']['success']
 assert case['methods']['ours']['success']
 for col, panel in enumerate(['start','frozen','adajepa','ours','goal']):
  ax=axes[row,col]
  raw=Image.open(D/case['frames'][panel]['path'])
  ax.imshow(raw.crop(case['display_crop_xyxy']),interpolation='none')
  ax.set_xticks([]);ax.set_yticks([])
  for sp in ax.spines.values():
   sp.set_visible(True);sp.set_color('#dce1e5');sp.set_linewidth(.6)
  if panel in methods:
   result=case['methods'][panel]
   outcome='Success' if result['success'] else 'Failure'
   ax.text(.5,-.06,f"{outcome} / {result['n_steps']} replans",transform=ax.transAxes,
           ha='center',va='top',fontsize=6.5,color=TEXT,
           fontweight='bold' if panel=='ours' else 'normal')
  if row==0:
   title=['Start','Frozen final','AdaJEPA final','SE-JEPA final','Goal image'][col]
   ax.set_title(title,fontsize=8,pad=5,color=TEXT)
  if col==0:
   ax.set_ylabel(case['label'],fontsize=7.5,labelpad=7,color=TEXT)
fig.text(.54,.025,'Target: object pose in the goal image. The green marker is static.',
         ha='center',va='bottom',fontsize=6.5,color=TEXT)
save(fig,'qualitative')
if '--qualitative-only' in sys.argv:
 raise SystemExit(0)
# Full trajectories retain raw frames and vector labels/loss curves.
for k in ['shape','visual']:
 fig=plt.figure(figsize=(7,3.35));gs=fig.add_gridspec(4,7,height_ratios=[1,1,1,1.05],left=.075,right=.99,top=.91,bottom=.1,wspace=.07,hspace=.3)
 loss_grid=gs[3,:].subgridspec(1,3,wspace=.48)
 for r,m in enumerate(methods):
  c=cases[k]['methods'][m];n=c['n_steps']
  for col,t in enumerate([0,2,5,10,20,30,'goal']):
   ax=fig.add_subplot(gs[r,col]);frame(ax,k,m,'goal' if t=='goal' else min(t,n))
   if r==0:ax.set_title('Goal' if t=='goal' else f'Replan {t}',fontsize=7,pad=3)
   if col==0:ax.set_ylabel(names[m],fontsize=8,color=colors[m])
   if t!='goal' and t>n:ax.patch.set_alpha(.2); ax.text(.5,.02,'held',transform=ax.transAxes,ha='center',fontsize=6,color='#666')
  ax=fig.add_subplot(loss_grid[0,r]);ax.plot(c['steps'],c['loss'],color=colors[m],lw=1.4);ax.set_title(names[m],fontsize=7,pad=3);ax.set_xlabel('MPC replan',fontsize=7);ax.tick_params(labelsize=6);ax.grid(alpha=.18)
  if r==0:ax.set_ylabel('Latent MSE',fontsize=7)
 fig.suptitle('Unseen square' if k=='shape' else 'Red-agent shift',fontsize=11,fontweight='bold',color=TEXT,y=.99)
 save(fig,k+'_full')
# Equal domain weights, matching all-domain aggregate in the supplied summaries.
fig,axs=plt.subplots(2,1,figsize=(3.3,3.1),sharex=True)
for ax,k,title in zip(axs,['shape','visual'],['PushObj: seven shapes','PushT: seven visual conditions']):
 s=json.loads((D/(k+'_summary.json')).read_text()); doms=s['groups']['all']
 for m in methods:
  ys=np.mean([s['curves']['CEM'][d][m]['mean'] for d in doms],axis=0)
  ax.plot(np.arange(1,len(ys)+1),ys,label=names[m],color=colors[m],lw=2.0 if m=='ours' else 1.3,ls=METHOD_STYLES[m])
  print(k,m,'final',ys[-1])
 ax.set_title(title,loc='left',fontsize=8,fontweight='bold');ax.set_ylim(0,100);ax.set_yticks([0,50,100]);ax.set_ylabel('Success (%)');ax.grid(alpha=.16);ax.set_xlim(1,30)
axs[0].legend(loc='lower right',frameon=False,fontsize=6.8);axs[1].set_xlabel('MPC replans');fig.tight_layout(h_pad=.8);save(fig,'planning_efficiency')
