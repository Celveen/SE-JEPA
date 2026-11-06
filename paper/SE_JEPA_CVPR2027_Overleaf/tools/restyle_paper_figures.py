"""Restyle paper plots from the reporting snapshot without changing measurements.

Palette provenance: Drive visual/shape build_figures.py BAR_COLOR supplies SE-JEPA
blue (#7FB5E8) and frozen warm gray (#CFCAC2); METHOD_COLOR supplies AdaJEPA red
(#EC4436). GD uses source PLANNER_COLOR #708CC3 and CEM source blue #3268DD.
Green plot accents are intentionally omitted following the author's preference.
Raw simulator colors and the author-supplied architecture SVG are preserved.
Run after build_paper_assets.py, then run build_revision_assets.py.
"""
from pathlib import Path
import csv
import html
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_teaser import COLORS as TEASER_COLORS, PANELS as TEASER_PANELS, plot_teaser

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / 'data'
FIG = PAPER / 'figures'
METHOD_COLORS = {'frozen': '#CFCAC2', 'adajepa': '#EC4436', 'ours': '#7FB5E8'}
PLANNER_COLORS = {'GD': '#708CC3', 'CEM': '#3268DD'}
TEXT = '#263548'
LABELS = {'frozen': 'Frozen', 'adajepa': 'AdaJEPA', 'ours': 'SE-JEPA'}
METHOD_STYLES = {'frozen': '--', 'adajepa': '-.', 'ours': '-'}
METHODS = ['frozen', 'adajepa', 'ours']
# High-contrast variants for the single-column plots. Markers and dashes
# carry the same distinctions as hue in grayscale print.
LINE_STYLES = {
    'frozen': dict(color='#553A76', linestyle=(0, (4, 2.4)), marker='s',
                   markerfacecolor='white'),
    'adajepa': dict(color='#BB4810', linestyle='-.', marker='^',
                    markerfacecolor='white'),
    'ours': dict(color='#006FC1', linestyle='-', marker='o'),
}
PLANNER_LINE_STYLES = {
    'GD': dict(color='#BB4810', linestyle='--', marker='s', markerfacecolor='white'),
    'CEM': dict(color='#006FC1', linestyle='-', marker='o'),
}

def emphasize_ours(legend):
    for label in legend.get_texts():
        label.set_fontweight('bold' if label.get_text() == 'SE-JEPA' else 'normal')


def set_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 7,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'axes.edgecolor': '#68727B',
        'text.color': TEXT, 'axes.labelcolor': TEXT,
        'xtick.color': '#4E5964', 'ytick.color': '#4E5964',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=.035)
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', pad_inches=.035, dpi=300)
    plt.close(fig)


def clean_cell(value):
    return html.unescape(re.sub(r'<[^>]*>', '', re.sub(r'<sup>.*?</sup>', '', value))).strip()


def final_main_rows(snapshot):
    """Use typeset main-table cells so a later audit propagates to the teaser.

    Falls back to the frozen reporting snapshot for absent rows. We do not edit
    either source or infer any corrected value in this style-only script.
    """
    rows = {clean_cell(r[0]): [clean_cell(c) for c in r] for r in snapshot['main_rows_html']}
    for line in (PAPER / 'tables/main_rows.tex').read_text().splitlines():
        if '&' not in line:
            continue
        cells = [c.strip() for c in line.split('&')]
        label = cells[0]
        key = next((k for k in ['DINO-WM', 'LeWM', 'AdaJEPA', 'Frozen'] if label.startswith(k)), None)
        if label.startswith(r'\ours'):
            key = 'SE-JEPA (Ours)'
        if key is None:
            continue
        values = []
        # The teaser compares runs in this study; the first numeric cell is the published protocol.
        for c in cells[2:]:
            m = re.search(r'(?<![A-Za-z])\d+(?:\.\d+)?', c)
            values.append(m.group(0) if m else '—')
        rows[key] = [key] + values
    return rows


def draw_teaser(s):
    values = final_main_rows(s)
    path = DATA / 'teaser_bars.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['method', *TEASER_PANELS])
        for method in TEASER_COLORS:
            row = values['SE-JEPA (Ours)' if method == 'SE-JEPA' else method]
            writer.writerow([method, row[1], row[2], row[4]])
    plot_teaser(path, FIG)


def draw_deployment(s, wide=False):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.12) if wide else (3.3, 1.62), constrained_layout=True)
    for ax, group, title in zip(axes, ['shape', 'visual'], ['(a) Unseen shapes', '(b) Visual conditions']):
        src = s['shape_rounds'] if group == 'shape' else s['visual_rounds']
        for planner, marker in [('GD', 's'), ('CEM', 'o')]:
            rows = [src[planner][str(i)]['unseen' if group == 'shape' else 'all'] for i in range(1,6)]
            ys=np.array([r['mean'] for r in rows]);sd=np.array([r.get('sd',0) for r in rows])
            color=PLANNER_COLORS[planner]
            ax.plot(range(1,6),ys,color=color,marker=marker,label=planner,markersize=3,lw=1.3)
            ax.fill_between(range(1,6),ys-sd,ys+sd,color=color,alpha=.14,lw=0)
        ax.set(title=title,xlabel='Round',xticks=range(1,6),ylim=(45,90))
        ax.set_title(title,fontsize=8)
        ax.tick_params(labelsize=7);ax.grid(axis='y',alpha=.16)
    axes[0].set_ylabel('Success (%)')
    axes[1].legend(frameon=False,loc='lower right',fontsize=6)
    save(fig,'deployment_wide' if wide else 'deployment')


def draw_deform(s):
    fig, axes=plt.subplots(1,2,figsize=(3.3,1.62),constrained_layout=True)
    for ax,task in zip(axes,['rope','granular']):
        a=np.array(s['deform'][task]['Ours_passes']['1e-4'])
        ax.plot(a[:,0],a[:,1],color=METHOD_COLORS['ours'],marker='o',markersize=3,lw=1.5,label='SE-JEPA')
        ax.fill_between(a[:,0],a[:,1]-a[:,2],a[:,1]+a[:,2],color=METHOD_COLORS['ours'],alpha=.20,lw=0)
        for name,key,ls in [('DINO-WM','frozen','--'),('AdaJEPA','adajepa','-.')]:
            ax.axhline(float(s['deform'][task][name]['cd']),color=METHOD_COLORS[key],ls=ls,lw=1.2,label=name)
        ax.set(title=task.capitalize(),xlabel='Pass',xticks=[1,2,3,4]);ax.tick_params(labelsize=7)
        ax.grid(axis='y',alpha=.16)
    axes[0].set_ylabel('Chamfer distance')
    axes[1].legend(frameon=False,fontsize=5.9,loc='center right',bbox_to_anchor=(1,.63))
    save(fig,'deform_passes')


def draw_visual_gains(s):
    fig,ax=plt.subplots(figsize=(6.8,1.85),constrained_layout=True)
    doms=['blur','snp1','dark','red_agent','red_object','red_goal']
    labels=['Blur','Noise','Dark','Red agent','Red object','Red goal']
    for off,planner in [(-.18,'CEM'),(.18,'GD')]:
        vals=[s['visual']['by_shape'][planner][d]['ours']['mean']-s['visual']['by_shape'][planner][d]['adajepa']['mean'] for d in doms]
        ax.bar(np.arange(6)+off,vals,.34,label=planner,color=PLANNER_COLORS[planner],zorder=3)
    ax.set_xticks(np.arange(6),labels);ax.set(ylabel='Gain over AdaJEPA (pp)',ylim=(0,40))
    ax.legend(frameon=False, ncol=2, loc='upper left', title='SE-JEPA',
              title_fontproperties={'weight': 'bold', 'size': 8}, alignment='left')
    ax.grid(axis='y',alpha=.16,zorder=0)
    save(fig,'visual_gains')


def draw_directions():
    if not (DATA/'cosine_gd.npy').exists():
        return
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.1),constrained_layout=True)
    fig.suptitle('SE-JEPA', fontweight='bold', fontsize=9)
    for ax,planner in zip(axes,['GD','CEM']):
        values=np.load(DATA/f'cosine_{planner.lower()}.npy')
        ax.hist(values,bins=np.linspace(-1,1,51),weights=np.ones(len(values))/len(values),
                color=PLANNER_COLORS[planner],edgecolor='white',linewidth=.3)
        ax.axvline(0,color='#777',ls=':',lw=.8)
        ax.set(title=f'{planner}: {len(values):,} adjacent updates',xlabel='Cosine similarity',ylabel='Fraction',xlim=(-1,1))
    save(fig,'update_directions')


def draw_planning_curves(s):
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.15),constrained_layout=True)
    for ax,key,title in zip(axes,['shape_full','visual'],['(a) PushObj held-out','(b) PushT held-out (seven)']):
        curves=s[key]['curves']['CEM'];doms=s[key]['groups']['all']
        for method in METHODS:
            y=np.mean([np.asarray(curves[d][method]['mean']) for d in doms],axis=0)
            ax.plot(np.arange(1,len(y)+1),y,label=LABELS[method],color=METHOD_COLORS[method],ls=METHOD_STYLES[method],lw=1.7)
        ax.set(title=title,xlabel='MPC replans',ylabel='Success (%)',xlim=(1,30),ylim=(0,100))
        ax.grid(alpha=.15);ax.legend(frameon=False,loc='lower right')
    save(fig,'planning_curves')


def main():
    set_style()
    s=json.loads((DATA/'paper_snapshot.json').read_text())
    draw_teaser(s);draw_deployment(s);draw_deployment(s,wide=True)
    draw_deform(s);draw_visual_gains(s);draw_directions();draw_planning_curves(s)
    print('Rebuilt eight quantitative figures with the Drive blue/red/gray palette.')

if __name__=='__main__':
    main()
