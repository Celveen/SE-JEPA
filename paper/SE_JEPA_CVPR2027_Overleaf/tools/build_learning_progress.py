"""Plot deployment learning across task-trained ResNet and DINO-WM backbones.

The numerical archive includes sample SD across the three rigid-object chains
and standard errors for the 50 deformable instances. Curves show the means.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from restyle_paper_figures import LINE_STYLES, PLANNER_LINE_STYLES, set_style

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
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.25))
    fig.subplots_adjust(left=.155, right=.97, top=.81, bottom=.18,
                        hspace=.91, wspace=.43)
    for ax, group, title in zip(axes[0], ['shape', 'color'],
                                ['(a) PushObj unseen', '(b) PushT color shifts']):
        for planner in ['GD', 'CEM']:
            rows = data[group][planner]
            y = np.array([r['mean'] for r in rows])
            ax.plot([r['round'] for r in rows], y, ms=3.1, mew=.85, lw=1.45,
                    label=planner, **PLANNER_LINE_STYLES[planner])
        ax.set(xticks=range(1, 6), ylim=(35, 85), yticks=[40, 60, 80],
               xlabel='Deployment round')
        ax.set_title(title, fontsize=7.6, pad=5)
    axes[0, 0].set_ylabel('Success (%)', fontsize=7.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, 1.01),
               ncol=2, frameon=False, fontsize=7.2, handlelength=2, columnspacing=1.2)
    for ax, task, title in zip(axes[1], ['rope', 'granular'],
                               ['(c) Rope', '(d) Granular']):
        rows = data['deform'][task]['passes']
        y = np.array([r['mean'] for r in rows])
        ax.plot([r['pass'] for r in rows], y, ms=3.2, lw=1.6, label='SE-JEPA',
                **LINE_STYLES['ours'])
        # Stagger the line identifiers because the two Granular references nearly coincide.
        for name, key, marker_x in [('DINO-WM', 'frozen', 1.5), ('AdaJEPA', 'adajepa', 3.5)]:
            reference = data['deform'][task][name]
            ax.plot([.85, marker_x, 4.15], [reference] * 3, markevery=[1],
                    ms=3.5, mew=.85, lw=1.12, label=name, **LINE_STYLES[key])
        ax.set(xticks=[1, 2, 3, 4], xlim=(.85, 4.15), xlabel='Consolidation pass')
        ax.set_title(title, fontsize=7.6, pad=5)
        ax.set_ylim((.16, .70) if task == 'rope' else (.18, .33))
        ax.set_yticks([.2, .4, .6] if task == 'rope' else [.20, .25, .30])
    axes[1, 0].set_ylabel('Chamfer distance', fontsize=7.2)
    for ax in axes.flat:
        ax.tick_params(labelsize=7, length=2.5)
        ax.xaxis.label.set_size(7.1)
        ax.grid(axis='y', color='#E1E5E9', lw=.5, zorder=0)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.54, -.025),
               ncol=3, frameon=False, fontsize=7.2, handlelength=2, columnspacing=.86)
    fig.savefig(FIG / 'learning_progress.pdf', bbox_inches='tight', pad_inches=.035)
    fig.savefig(FIG / 'learning_progress.png', bbox_inches='tight', pad_inches=.035, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    values = load_progress()
    draw_progress(values)
    print('Built 5-round unseen/color and 4-pass deformable learning curves.')
    for planner in ['GD', 'CEM']:
        print(planner, 'color success:', [round(r['mean'], 4) for r in values['color'][planner]])
