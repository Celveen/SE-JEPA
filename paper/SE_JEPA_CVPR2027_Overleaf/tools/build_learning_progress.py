"""Plot deployment learning across task-trained ResNet and DINO-WM backbones.

Rigid-object bands: sample SD of three learning-chain success rates.
Deformable bands: archived standard errors over 50 repeated instances.
Round/pass indices are deployment consolidation, not pretraining epochs.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from restyle_paper_figures import METHOD_COLORS, PLANNER_COLORS, TEXT, set_style

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / 'data'
FIG = PAPER / 'figures'
COLORS = ['red_agent', 'red_object', 'red_goal']


def load_progress():
    snapshot_path = DATA / 'paper_snapshot.json'
    color_path = DATA / 'revision_assets/visual_color_learning_records.json'
    snapshot = json.loads(snapshot_path.read_text())
    records = json.loads(color_path.read_text())['records']
    output = {
        'sources': {str(p.relative_to(PAPER)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [snapshot_path, color_path]},
        'shape': {}, 'color': {}, 'deform': {},
        'protocol': {
            'shape': 'PushObj I, square, small tee; new targets each round; 150 episodes per chain and round; three chains.',
            'color': 'PushT red agent, red object, red goal; new targets each round; 150 episodes per chain and round; three chains.',
            'rigid_error': 'Sample standard deviation across chain-level rates (ddof=1).',
            'deform': 'Slow learning rate 1e-4; same 50 instances revisited for four passes; mean CD and standard error from the reporting snapshot.',
        },
    }
    for planner in ['GD', 'CEM']:
        output['shape'][planner] = []
        output['color'][planner] = []
        for rnd in range(1, 6):
            shape = snapshot['shape_rounds'][planner][str(rnd)]['unseen']
            assert len(shape['units']) == 3 and shape['episodes'] == 450
            assert np.isclose(np.mean(shape['units']), shape['mean'])
            assert np.isclose(np.std(shape['units'], ddof=1), shape['sd'])
            output['shape'][planner].append({'round': rnd, **shape})
            chains = []
            counts = []
            for chain in [1, 2, 3]:
                rows = [r for r in records if r['optimizer'] == planner
                        and r['round'] == rnd and r['chain'] == chain]
                assert len(rows) == 3 and {r['tag'] for r in rows} == set(COLORS)
                assert all(r['n'] == 50 and r['paired_targets_verified'] for r in rows)
                n = sum(r['n'] for r in rows)
                successes = sum(r['successes'] for r in rows)
                chains.append(100 * successes / n)
                counts.append({'chain': chain, 'successes': successes, 'episodes': n})
            output['color'][planner].append({
                'round': rnd, 'mean': float(np.mean(chains)),
                'sd': float(np.std(chains, ddof=1)), 'units': chains,
                'episodes': 450, 'counts': counts,
            })
    for task in ['rope', 'granular']:
        rows = snapshot['deform'][task]['Ours_passes']['1e-4']
        assert [r[0] for r in rows] == [1, 2, 3, 4]
        output['deform'][task] = {
            'passes': [{'pass': r[0], 'mean': r[1], 'se': r[2]} for r in rows],
            'DINO-WM': float(snapshot['deform'][task]['DINO-WM']['cd']),
            'AdaJEPA': float(snapshot['deform'][task]['AdaJEPA']['cd']),
        }
    (DATA / 'learning_progress.json').write_text(json.dumps(output, indent=2) + '\n')
    return output


def draw_progress(data):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.20))
    fig.subplots_adjust(left=.155, right=.98, top=.935, bottom=.17,
                        hspace=.67, wspace=.43)
    for ax, group, title in zip(axes[0], ['shape', 'color'],
                                ['(a) PushObj unseen', '(b) PushT color shifts']):
        for planner, marker in [('GD', 's'), ('CEM', 'o')]:
            rows = data[group][planner]
            y = np.array([r['mean'] for r in rows])
            sd = np.array([r['sd'] for r in rows])
            ax.plot(range(1, 6), y, marker=marker, color=PLANNER_COLORS[planner],
                    ms=3, lw=1.35, label=planner)
            ax.fill_between(range(1, 6), y - sd, y + sd,
                            color=PLANNER_COLORS[planner], alpha=.15, lw=0)
        ax.set(xticks=range(1, 6), ylim=(35, 85), yticks=[40, 60, 80],
               xlabel='Deployment round')
        ax.set_title(title, fontsize=7.1, pad=5)
    axes[0, 0].set_ylabel('Success (%)', fontsize=7)
    axes[0, 0].legend(loc='lower left', ncol=2, frameon=False, fontsize=5.9,
                      handlelength=1.1, columnspacing=.65, borderaxespad=.15)
    for ax, task, title in zip(axes[1], ['rope', 'granular'],
                               ['(c) Rope', '(d) Granular']):
        rows = data['deform'][task]['passes']
        y = np.array([r['mean'] for r in rows])
        se = np.array([r['se'] for r in rows])
        ax.plot(range(1, 5), y, color=METHOD_COLORS['ours'], marker='o',
                ms=3, lw=1.7, label='SE-JEPA')
        ax.fill_between(range(1, 5), y - se, y + se,
                        color=METHOD_COLORS['ours'], alpha=.20, lw=0)
        for name, key, style in [('DINO-WM', 'frozen', '--'), ('AdaJEPA', 'adajepa', '-.')]:
            ax.axhline(data['deform'][task][name], color=METHOD_COLORS[key],
                       ls=style, lw=1.1, label=name)
        ax.set(xticks=[1, 2, 3, 4], xlabel='Consolidation pass')
        ax.set_title(title, fontsize=7.1, pad=5)
        ax.set_ylim((.16, .70) if task == 'rope' else (.18, .33))
        ax.set_yticks([.2, .4, .6] if task == 'rope' else [.20, .25, .30])
    axes[1, 0].set_ylabel('Chamfer distance', fontsize=7)
    for ax in axes.flat:
        ax.tick_params(labelsize=6.5, length=2.5)
        ax.xaxis.label.set_size(6.8)
        ax.grid(axis='y', color='#E1E5E9', lw=.5, zorder=0)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.55, -.025),
               ncol=3, frameon=False, fontsize=6, handlelength=1.4, columnspacing=.7)
    fig.savefig(FIG / 'learning_progress.pdf', bbox_inches='tight', pad_inches=.035)
    fig.savefig(FIG / 'learning_progress.png', bbox_inches='tight', pad_inches=.035, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    values = load_progress()
    draw_progress(values)
    print('Built 5-round unseen/color and 4-pass deformable learning curves.')
    for planner in ['GD', 'CEM']:
        print(planner, 'color success:', [round(r['mean'], 4) for r in values['color'][planner]])
