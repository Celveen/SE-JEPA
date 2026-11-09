#!/usr/bin/env python3
"""Replay four archived GD visual episodes on their exact shared target.

Uses original simulator source, recorded normalized actions, and archived states.
No model weights, new planning, image synthesis, or pixel modification is used.
All file references required to rebuild this package are relative to this file.
"""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'mpl-cache'))
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_simulator():
    path = ROOT/'source/pusht_env.py'
    original = path.read_text()
    tree = ast.parse(original, filename=str(path))
    skipped, keep = [], []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(x.name in ['torch', 'einops', 'skimage.transform'] for x in node.names):
            skipped.append(ast.get_source_segment(original, node))
        else:
            keep.append(node)
    tree.body = keep
    module = types.ModuleType('original_pusht_env')
    module.__file__ = str(path)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module.PushTEnv, skipped


def normalization():
    values = {}
    for node in ast.parse((ROOT/'source/pusht_dset.py').read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ['ACTION_MEAN', 'ACTION_STD']:
                values[name] = np.asarray(ast.literal_eval(node.value.args[0]), dtype=np.float32)
    return values['ACTION_MEAN'], values['ACTION_STD']


def replay(Env, initial, normalized_actions, seed, mean, std):
    env = Env(shape='T', color='Red', with_velocity=True, with_target=True)
    env.seed(seed)
    env.reset_to_state = initial.copy()
    obs, state = env.reset()
    frames, states = [obs['visual']], [state]
    physical = normalized_actions.reshape(-1, 2).astype(np.float32)*std+mean
    for action in physical:
        obs, _, _, info = env.step(action)
        frames.append(obs['visual'])
        states.append(info['state'])
    env.close()
    return np.asarray(frames), np.asarray(states), physical


def goal_image(Env, state, seed):
    # Render a static target with zero pusher velocity, as in the archived
    # visual reference. This keeps the recorded target pose fixed during the
    # setup substep. Raw target states and outcome evaluation are not modified.
    env = Env(shape='T', color='Red', with_velocity=True, with_target=True)
    render_state = state.copy(); render_state[5:] = 0
    env.seed(seed); env.reset_to_state = render_state
    obs, rendered_state = env.reset()
    image = obs['visual']
    env.close()
    assert np.array_equal(rendered_state[:5], state[:5])
    return image


def success_metrics(states, goal):
    pos = np.linalg.norm(states[:, :4]-goal[:4], axis=1)
    angle = np.abs((states[:, 4]-goal[4]+np.pi) % (2*np.pi)-np.pi)
    return (pos < 20) & (angle < np.pi/9), pos, angle


def main():
    inputs = json.loads((ROOT/'inputs.json').read_text())
    Env, skipped = load_simulator()
    mean, std = normalization()

    # Independent control: the current paper's known, archived visual episode.
    control = np.load(ROOT/'control/frozen_trajectory.npz')
    cf, cs, _ = replay(Env, control['states'][0], control['actions'], 2000, mean, std)
    control_goal = goal_image(Env, control['goal'], 2000)
    control_validation = {
        'all_31_model_slot_states_identical': bool(np.array_equal(cs[::5], control['states'])),
        'state_max_abs_error': float(np.max(abs(cs[::5]-control['states']))),
        'start_pixels_identical': bool(np.array_equal(cf[0], np.asarray(Image.open(ROOT/'control/start.png')))),
        'final_pixels_identical': bool(np.array_equal(cf[-1], np.asarray(Image.open(ROOT/'control/frozen.png')))),
        'goal_pixels_identical': bool(np.array_equal(control_goal, np.asarray(Image.open(ROOT/'control/goal.png')))),
        'reference_scope': 'The existing visual archive contains all 31 states and start/final/goal PNGs, but not reference images for every intermediate state.',
    }
    assert all(control_validation[k] for k in ['all_31_model_slot_states_identical', 'start_pixels_identical', 'final_pixels_identical', 'goal_pixels_identical']), control_validation

    method_results = {}
    common_start = common_goal = common_start_image = common_goal_image = None
    selected_frames = []
    for method in ['frozen', 'dinowm', 'adajepa', 'ours']:
        spec = inputs['methods'][method]
        npz = np.load(ROOT/'raw'/method/'final_eval_results.npz')
        idx = spec['sample_idx']
        initial, goal = npz['state_0'][idx], npz['state_g'][idx]
        if common_start is None:
            common_start, common_goal = initial.copy(), goal.copy()
        assert np.array_equal(initial, common_start) and np.array_equal(goal, common_goal)
        recorded_length = float(npz['action_len'][idx])
        slots = len(npz['actions'][idx]) if np.isinf(recorded_length) else int(recorded_length)
        actions = npz['actions'][idx, :slots]
        assert actions.shape[-1] == 10
        frames, states, physical = replay(Env, initial, actions, spec['seed'], mean, std)
        goal_frame = goal_image(Env, goal, spec['seed'])
        if common_start_image is None:
            common_start_image, common_goal_image = frames[0].copy(), goal_frame.copy()
        assert np.array_equal(frames[0], common_start_image) and np.array_equal(goal_frame, common_goal_image)
        per_replan = spec['physical_steps_per_mpc_replan']
        assert len(physical) % per_replan == 0
        success, pos, angle = success_metrics(states, goal)
        checkpoints = np.arange(0, len(states), per_replan)
        recorded_success = bool(npz['successes'][idx])
        assert bool(success[-1]) == recorded_success, (method, 'recorded/replay outcome mismatch')
        output = ROOT/'frames'/method; output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frames[0]).save(output/'start.png')
        Image.fromarray(frames[-1]).save(output/'final.png')
        Image.fromarray(goal_frame).save(output/'goal.png')
        for step in range(0, len(frames), 5):
            Image.fromarray(frames[step]).save(output/f'env_step_{step:03d}.png')
        np.savez_compressed(output/'trajectory.npz', visual=frames, states=states,
                            normalized_actions=actions, physical_actions=physical,
                            initial=initial, goal=goal)
        selected_frames.extend([frames[0], frames[-1], goal_frame])
        method_results[method] = {
            **spec,
            'recorded_action_len_model_slots': 'inf' if np.isinf(recorded_length) else slots,
            'executed_model_slots': slots,
            'executed_physical_steps': len(physical),
            'executed_mpc_replans': len(physical)//per_replan,
            'trailing_padding_not_executed': len(npz['actions'][idx])-slots,
            'archived_success': recorded_success,
            'replay_final_success': bool(success[-1]),
            'replay_success_at_mpc_boundaries': success[checkpoints].tolist(),
            'final_position_distance': float(pos[-1]),
            'final_angle_distance_rad': float(angle[-1]),
            'final_state': states[-1].tolist(),
            'final_frame': str((output/'final.png').relative_to(ROOT)),
            'final_frame_sha256': sha(output/'final.png'),
        }

    # A common view box for the new scene, preserving every colored element.
    mask = np.any([im.max(axis=-1)-im.min(axis=-1) > 5 for im in selected_frames], axis=0)
    yy, xx = np.nonzero(mask)
    # Keep a small margin from the static outer canvas border.
    x0 = int(np.clip((xx.min()+xx.max()+1)//2-84, 5, 51))
    y0 = int(np.clip((yy.min()+yy.max()+1)//2-84, 5, 51))
    crop = [x0, y0, x0+168, y0+168]
    assert xx.min() >= crop[0] and xx.max() < crop[2] and yy.min() >= crop[1] and yy.max() < crop[3]
    Image.fromarray(common_start_image).save(ROOT/'start.png')
    Image.fromarray(common_goal_image).save(ROOT/'goal.png')
    record = {
        'selection': inputs['selection'],
        'source_simulator_commit': inputs['source_simulator_commit'],
        'simulator_physics_and_rendering_edited': False,
        'unused_optional_imports_skipped_at_load': skipped,
        'normalization': {'mean': mean.tolist(), 'std': std.tolist(), 'formula': 'Reshape each recorded 10D model slot into five 2D physical actions, then multiply by std and add mean. The original simulator scales by 100 and applies relative displacement.'},
        'physical_steps_per_model_slot': 5,
        'physics_substeps_per_physical_step': 10,
        'common_budget_physical_steps': 150,
        'common_initial_state': common_start.tolist(),
        'common_goal_state': common_goal.tolist(),
        'four_initial_and_goal_states_exactly_equal': True,
        'four_start_and_goal_frames_pixel_identical': True,
        'display_crop_xyxy': crop,
        'display_crop_preserves_all_colored_foreground': True,
        'control_validation': control_validation,
        'methods': method_results,
        'goal_rendering': 'Static target illustration: the archived target pose is unchanged, and pusher velocity is zero only for image rendering so reset does not advance the pusher. This reproduces the existing archived visual goal PNG pixel-for-pixel. All raw seven-dimensional goal states are preserved; success evaluation uses their original values.',
        'versions': {k: importlib.metadata.version(k) for k in ['numpy','pymunk','pygame','opencv-python','shapely','gym']},
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'source').glob('*') if p.is_file()},
    }
    (ROOT/'replay_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    # Drop-in fragment for data/revision_assets/selection_manifest.json.
    prefix = 'dinowm_visual_mixed'
    frames_fragment = {}
    for panel, file in {'start':'start.png', 'goal':'goal.png',
                        **{m:f'frames/{m}/final.png' for m in method_results}}.items():
        frames_fragment[panel] = {'path':f'{prefix}/{file}', 'sha256':sha(ROOT/file),
                                  'frames_origin':'Original simulator rendering/action replay; source and validation in replay_provenance.json'}
    methods_fragment = {}
    for method, info in method_results.items():
        trajectory = ROOT/'frames'/method/'trajectory.npz'
        methods_fragment[method] = {
            'success':info['replay_final_success'],
            'n_steps':info['executed_mpc_replans'],
            'physical_steps':info['executed_physical_steps'],
            'source_sample_idx':info['sample_idx'],
            'source_seed':info['seed'],
            'source':info['original_source_directory'],
            'source_npz_sha256':info['source_npz_sha256'],
            'config_sha256':info['config_sha256'],
            'frames_origin':'Original simulator replay of the recorded executed actions, truncated at the archived action_len',
            'trajectory':{'path':f'{prefix}/frames/{method}/trajectory.npz','sha256':sha(trajectory)},
            'provenance':f'{prefix}/replay_provenance.json',
            'final_state':info['final_state'],
        }
    fragment = {
        'id':'visual_mixed14_GD_red_object', 'task':'PushT', 'domain':'red_object',
        'planner':'GD', 'label':'Red object (GD)',
        'protocol':'Selected mixed14 online target; DINO-WM has the same state pair in its separate visual evaluation',
        'selection':inputs['selection'], 'original_image_size':[224,224],
        'display_crop_xyxy':crop, 'frames':frames_fragment, 'methods':methods_fragment,
        'validation':{'step_00.png_identical_across_methods':True,'goal.png_identical_across_methods':True,
                      'crop_preserves_all_colored_foreground':True,
                      'raw_initial_and_goal_states_identical_across_methods':True,
                      'simulator_checked_against_existing_visual_frozen_trajectory':True},
    }
    (ROOT/'selection_fragment.json').write_text(json.dumps(fragment,indent=2)+'\n')
    print(json.dumps({'control_validation': control_validation, 'display_crop':crop,
                     'methods': {k:{x:v[x] for x in ['executed_physical_steps','executed_mpc_replans','replay_final_success','final_position_distance','final_angle_distance_rad']} for k,v in method_results.items()}},indent=2))


if __name__ == '__main__':
    main()
