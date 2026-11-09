"""Replay archived actions in the repository's original PyMunk simulator.

No checkpoint, model inference, new action search, or image synthesis is used.
The native simulator source is retained verbatim in source/. Only unused optional
imports (torch, einops, skimage.transform) are skipped when loading that module.
"""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import types

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / '.mpl-cache'))

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[2]
CASE = PAPER/'data/revision_assets/selected_cases/shape_GD_I'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def simulator():
    path = ROOT/'source/pusht_env.py'
    original = path.read_text()
    tree = ast.parse(original, filename=str(path))
    skipped = []
    keep = []
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

def norm_constants():
    tree = ast.parse((ROOT/'source/pusht_dset.py').read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0],ast.Name):
            name = node.targets[0].id
            if name in ['ACTION_MEAN','ACTION_STD']:
                out[name] = np.asarray(ast.literal_eval(node.value.args[0]),dtype=np.float32)
    return out['ACTION_MEAN'],out['ACTION_STD']

def replay(Env, initial, actions, mean, std):
    env = Env(shape='I', with_velocity=True, with_target=True)
    env.seed(2000)
    env.reset_to_state = initial.copy()
    obs,state = env.reset()
    frames, states = [obs['visual']], [state]
    physical = actions.reshape(-1,2).astype(np.float32)*std+mean
    for i, action in enumerate(physical):
        obs, _, _, info = env.step(action)
        if (i+1)%5 == 0:
            frames.append(obs['visual'])
            states.append(info['state'])
    env.close()
    return np.asarray(frames),np.asarray(states),physical

def main():
    Env, skipped = simulator()
    mean,std = norm_constants()
    frozen = np.load(CASE/'frozen_trajectory.npz')
    ff,fs,fa = replay(Env,frozen['states'][0],frozen['actions'],mean,std)
    validation = {
        'frozen_all_31_frames_pixel_identical':bool(np.array_equal(ff,frozen['visual'])),
        'frozen_all_31_states_exact':bool(np.array_equal(fs,frozen['states'])),
        'frozen_state_max_abs_error':float(np.max(abs(fs-frozen['states']))),
        'frozen_per_frame_different_pixels':[int(np.any(x!=y,axis=-1).sum()) for x,y in zip(ff,frozen['visual'])],
    }
    dino = np.load(ROOT/'dinowm-I-H7t3-GD.npz')
    initial,goal,actions = dino['state_0'][0],dino['state_g'][0],dino['actions'][0]
    frames,states,physical = replay(Env,initial,actions,mean,std)
    # Render the archived target state itself. reset() advances one physics
    # substep, so restore all recorded state components and refresh the shape
    # transforms before taking this static goal image. No velocity is discarded.
    env=Env(shape='I',with_velocity=True,with_target=True)
    env.seed(2000);env.reset_to_state=goal.copy();reset_goal,reset_goal_state=env.reset()
    env.agent.position=goal[:2].tolist();env.agent.velocity=goal[5:].tolist()
    env.block.angle=float(goal[4]);env.block.position=goal[2:4].tolist()
    env.space.reindex_shapes_for_body(env.agent);env.space.reindex_shapes_for_body(env.block)
    go={'visual':env.render('rgb_array')};gs=env._get_obs();env.close()
    start_original=np.asarray(Image.open(CASE/'start.png'))
    goal_original=np.asarray(Image.open(CASE/'goal.png'))
    original_frozen=np.load(ROOT/'official_frozen_I_original.npz')
    validation.update({
        'dino_all_50_initial_states_equal_official_frozen':bool(np.array_equal(dino['state_0'],original_frozen['state_0'])),
        'dino_all_50_goal_states_equal_official_frozen':bool(np.array_equal(dino['state_g'],original_frozen['state_g'])),
        'official_frozen_actions_equal_archived_figure_trajectory':bool(np.array_equal(original_frozen['actions'][0],frozen['actions'])),
        'dino_initial_state_equals_existing_case':bool(np.array_equal(initial,frozen['states'][0])),
        'dino_start_pixels_equal_existing_case':bool(np.array_equal(frames[0],start_original)),
        'dino_start_different_pixels':int(np.any(frames[0]!=start_original,axis=-1).sum()),
        'dino_goal_pixels_equal_existing_case':bool(np.array_equal(go['visual'],goal_original)),
        'dino_goal_different_pixels':int(np.any(go['visual']!=goal_original,axis=-1).sum()),
        'dino_goal_render_state_exact':bool(np.array_equal(gs,goal)),
        'goal_reset_before_pose_restore_different_pixels':int(np.any(reset_goal['visual']!=goal_original,axis=-1).sum()),
    })
    # The figure's existing display crop must also preserve the new final frame.
    m=frames[-1].max(axis=-1)-frames[-1].min(axis=-1)>5
    outside=m.copy();outside[0:168,50:218]=False
    validation['dino_final_existing_crop_preserves_all_colored_foreground']=not bool(outside.any())
    assert all(validation[k] for k in [
        'frozen_all_31_frames_pixel_identical','frozen_all_31_states_exact',
        'dino_all_50_initial_states_equal_official_frozen','dino_all_50_goal_states_equal_official_frozen',
        'official_frozen_actions_equal_archived_figure_trajectory',
        'dino_start_pixels_equal_existing_case','dino_goal_pixels_equal_existing_case','dino_goal_render_state_exact'
    ]), 'Replay not validated for use'
    posdiff=np.linalg.norm(states[:,:4]-goal[:4],axis=1)
    anglediff=abs((states[:,4]-goal[4]+np.pi)%(2*np.pi)-np.pi)
    success=(posdiff<20)&(anglediff<np.pi/9)
    out=ROOT/'frames';out.mkdir(exist_ok=True)
    for i,frame in enumerate(frames):Image.fromarray(frame).save(out/f'step_{i:02d}.png')
    Image.fromarray(go['visual']).save(out/'goal.png')
    np.savez_compressed(ROOT/'dinowm_I_GD_replay.npz',visual=frames,states=states,physical_actions=physical,normalized_actions=actions,goal=goal,goal_render_state=gs)
    record={
        'source_commit':'eec0ba00fbb945c4dc70bd4804035f845b67a463',
        'run_path':'research_snapshots/2026-11-07/completed_runs/CL-JEPA-cudagraph/baselines/dinowm_H7t3_GD_bias16k_reference_e3_20261106/chunk_00_I',
        'seed':2000,'sample_idx':0,'planner':'GD','checkpoint':'dinowm_pushobj_bias16k_reference_e3, epoch 3',
        'archived_success':bool(dino['successes'][0]),'archived_action_len':'inf' if np.isinf(dino['action_len'][0]) else float(dino['action_len'][0]),
        'physical_steps':len(physical),'mpc_replans':len(actions),'physics_substeps_per_physical_step':10,
        'action_decoding':'reshape [30,10] -> [150,2], a = normalized_action * ACTION_STD + ACTION_MEAN; original simulator multiplies a by 100 and applies relative displacements',
        'action_mean':mean.tolist(),'action_std':std.tolist(),
        'goal_rendering':'Render the recorded target pose with its original velocity retained, after restoring the recorded state and reindexing shapes. This avoids the automatic one-substep advance in reset(), which shifts the pusher in a static goal illustration. The resulting full frame is pixel-identical to the archived Figure 3 goal image.',
        'official_frozen_validation_run':'research_snapshots/2026-11-07/completed_runs/CL-JEPA-cudagraph/cmp_gd_frozen_official/w0/chunk_00_I/final_eval_results.npz',
        'runtime_imports_skipped':skipped,'simulator_physics_and_rendering_edited':False,
        'validation':validation,
        'replay_success_at_each_replan':success.tolist(),
        'replay_final_success':bool(success[-1]),
        'replay_final_position_distance':float(posdiff[-1]),'replay_final_angle_distance_rad':float(anglediff[-1]),
        'initial_state':initial.tolist(),'goal_state':goal.tolist(),'goal_render_state':gs.tolist(),'final_state':states[-1].tolist(),
        'versions':{k:importlib.metadata.version(k) for k in ['numpy','pymunk','pygame','opencv-python','shapely','gym']},
        'hashes':{str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'source').glob('*'))+[ROOT/'dinowm-I-H7t3-GD.npz',ROOT/'dinowm-I-H7t3-GD-config.yaml',ROOT/'official_frozen_I_original.npz',ROOT/'dinowm_I_GD_replay.npz']+list(out.glob('*.png'))},
    }
    (ROOT/'replay_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['validation','replay_final_success','replay_final_position_distance','replay_final_angle_distance_rad']},indent=2))

if __name__=='__main__':main()
