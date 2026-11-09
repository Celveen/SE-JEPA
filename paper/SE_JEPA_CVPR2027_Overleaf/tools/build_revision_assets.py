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
from restyle_deform_rollouts import draw_deform_rollouts
from restyle_paper_figures import LINE_STYLES, TEXT, emphasize_ours, set_style
from PIL import Image
P=Path(__file__).resolve().parents[1]; D=P/'data/revision_assets'; F=P/'figures'
set_style()
methods=['frozen','adajepa','ours']; names={'frozen':'Frozen','adajepa':'AdaJEPA','ours':'SE-JEPA'}
cases={k:json.loads((D/(k+'_case.json')).read_text()) for k in ['shape','visual']}
def frame_border(ax, emphasized):
 for spine in ax.spines.values():
  spine.set_visible(True)
  spine.set_color(TEXT if emphasized else '#dce1e5')
  spine.set_linewidth(1.35 if emphasized else .55)

def frame(ax,k,m,step):
 f=D/k/m/(('goal' if step=='goal' else f'step_{step:02d}')+'.png')
 ax.imshow(Image.open(f),interpolation='none');ax.set_xticks([]);ax.set_yticks([])
 frame_border(ax, m == 'ours' and step not in (0, 'goal'))
def save(fig,n):
 fig.savefig(F/(n+'.pdf'),bbox_inches='tight',pad_inches=.025)
 fig.savefig(F/(n+'.png'),bbox_inches='tight',pad_inches=.025,dpi=300)
 plt.close(fig)

def draw_planning_efficiency():
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.12), sharex=True)
    fig.subplots_adjust(left=.17, right=.97, bottom=.13, top=.82, hspace=.58)
    for ax, key, title in zip(axes, ['shape', 'visual'],
                              ['PushObj: seven shapes', 'PushT: seven visual conditions']):
        summary = json.loads((D / f'{key}_summary.json').read_text())
        domains = summary['groups']['all']
        # Each domain has equal weight, matching the reported aggregate.
        # Frozen and AdaJEPA use the released task-specific checkpoints;
        # see drive_reference/shape_readme.txt and visual_protocol.json.
        for offset, method in enumerate(methods):
            y = np.mean([summary['curves']['GD'][d][method]['mean'] for d in domains], axis=0)
            ax.plot(np.arange(1, len(y) + 1), y, label=names[method],
                    lw=2.15 if method == 'ours' else 1.12,
                    ms=3.8 if method == 'ours' else 3.1, mew=.85,
                    markevery=(offset * 2, 6), **LINE_STYLES[method])
        ax.set_title(title, loc='left', fontsize=8, pad=5)
        ax.set(xlim=(1, 30), ylim=(0, 100), yticks=[0, 50, 100], ylabel='Success (%)')
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', color='#E1E5E9', lw=.5)
    axes[1].set(xlabel='MPC replans', xticks=[1, 5, 10, 15, 20, 25, 30])
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, 1.03),
                        ncol=3, fontsize=7.2, frameon=False, handlelength=2, columnspacing=1.05)
    emphasize_ours(legend)
    save(fig, 'planning_efficiency')

if '--planning-only' in sys.argv:
    draw_planning_efficiency()
    raise SystemExit(0)

# Selected failures/successes share pixel-identical initial and goal observations.
# Equal-size crops enlarge the raw simulator frames without changing their pixels.
selection=json.loads((D/'selection_manifest.json').read_text())['cases']
fig = plt.figure(figsize=(7, 2.55))
rows = fig.add_gridspec(len(selection), 1, left=.085, right=.995,
                       top=.87, bottom=.12, hspace=.68)
panel_titles={'start':'Start','frozen':'Frozen final','dinowm':'DINO-WM final',
              'adajepa':'AdaJEPA final','ours':'SE-JEPA final','goal':'Goal image'}
for row, case in enumerate(selection):
 assert not case['methods']['frozen']['success']
 assert not case['methods']['adajepa']['success']
 assert case['methods']['ours']['success']
 panels=['start','frozen'] + (['dinowm'] if 'dinowm' in case['methods'] else []) + ['adajepa','ours','goal']
 cells=rows[row].subgridspec(1,len(panels),wspace=.04)
 for col, panel in enumerate(panels):
  ax=fig.add_subplot(cells[col])
  raw=Image.open(D/case['frames'][panel]['path'])
  ax.imshow(raw.crop(case['display_crop_xyxy']),interpolation='none')
  ax.set_xticks([]);ax.set_yticks([])
  frame_border(ax, panel == 'ours')
  if panel in case['methods']:
   result=case['methods'][panel]
   outcome='Success' if result['success'] else 'Failure'
   ax.text(.5,-.06,f"{outcome} / {result['physical_steps']} steps",transform=ax.transAxes,
           ha='center',va='top',fontsize=6.5,color=TEXT,
           fontweight='bold' if panel=='ours' else 'normal')
  ax.set_title(panel_titles[panel],fontsize=8,pad=5,color=TEXT,
               fontweight='bold' if panel=='ours' else 'normal')
  if col==0:
   ax.set_ylabel(case['label'],fontsize=7.5,labelpad=7,color=TEXT)
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
   if col==0:ax.set_ylabel(names[m],fontsize=8,color=TEXT,fontweight='bold' if m=='ours' else 'normal')
   if t!='goal' and t>n:ax.patch.set_alpha(.2); ax.text(.5,.02,'held',transform=ax.transAxes,ha='center',fontsize=6,color='#666')
  ax=fig.add_subplot(loss_grid[0,r])
  ax.plot(c['steps'], c['loss'], lw=2.15 if m=='ours' else 1.1,
          ms=3.2 if m=='ours' else 2.5, mew=.75,
          markevery=max(1, len(c['steps']) // 5), **LINE_STYLES[m])
  ax.set_title(names[m],fontsize=7,pad=3,fontweight='bold' if m=='ours' else 'normal')
  ax.set_xlabel('MPC replan',fontsize=7);ax.tick_params(labelsize=6);ax.grid(alpha=.18)
  if r==0:ax.set_ylabel('Latent MSE',fontsize=7)
 fig.suptitle('Unseen square' if k=='shape' else 'Red-agent shift',fontsize=11,color=TEXT,y=.99)
 save(fig,k+'_full')
draw_deform_rollouts(F / 'source/deform_qualitative_original.pdf', F / 'deform_qualitative.pdf')
if '--rollouts-only' in sys.argv:
 raise SystemExit(0)
draw_planning_efficiency()
