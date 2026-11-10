"""Verify the epoch-three random-data PushT DINO-WM CEM replacement."""
import hashlib,json
from pathlib import Path
import numpy as np

def verified_pusht_cem(directory):
    source=json.loads((directory/'sources.json').read_text())
    for entry in source['files']:
        assert hashlib.sha256((directory/entry['local_file']).read_bytes()).hexdigest()==entry['sha256']
    summary=json.loads((directory/'pipeline_summary.json').read_text())
    checkpoint=json.loads((directory/'evaluation_checkpoint.json').read_text())
    assert summary['epoch']==checkpoint['epoch']==3
    assert summary['checkpoint_sha256']==checkpoint['sha256']=='4cba2153c3851dd4f9fabc7ad67fd8384df11fc65422bf9d9dfc7000d48fa3a4'
    assert summary['training_shapes']==['T']
    controller=summary['controller'];assert controller['max_replans']==6 and controller['physical_steps_per_replan']==25 and controller['max_physical_steps']==150
    conditions=summary['by_planner']['cem']['conditions']
    for domain in source['raw_episode_files_available']:
        raw=directory/'raw'/domain
        protocol=json.loads((raw/'runtime_protocol.json').read_text())
        weights=json.loads((raw/'frozen_weights_check.json').read_text())
        assert protocol['planner']=='cem' and protocol['model_epoch']==3 and protocol['frozen_model']
        assert protocol['max_replans']==6 and protocol['executed_physical_steps_per_replan']==25
        assert weights['unchanged'] and weights['before_sha256']==weights['after_sha256']
        with np.load(raw/'final_eval_results.npz') as data:
            successes=data['successes'];slots=data['action_len']
            assert len(successes)==conditions[domain]['episodes']==50
            assert int(successes.sum())==conditions[domain]['successes']
            replans=np.where(successes,slots/5,6)
            assert abs(replans.mean()-conditions[domain]['mean_mpc_replans'])<1e-10
            assert np.all(replans<=6) and np.all(replans>=0)
    shifts=['blur','snp','dark','red_agent','red_object','red_goal'];colors=shifts[-3:]
    n=sum(conditions[x]['episodes'] for x in shifts);k=sum(conditions[x]['successes'] for x in shifts)
    nc=sum(conditions[x]['episodes'] for x in colors);kc=sum(conditions[x]['successes'] for x in colors)
    total=sum(conditions[x]['episodes']*conditions[x]['mean_mpc_replans'] for x in shifts)
    result={'shift_rate':100*k/n,'color_rate':100*kc/nc,'steps':total/n,'shift_successes':k,'shift_episodes':n,'color_successes':kc,'color_episodes':nc,'mpc_replans_total':int(round(total)),'environment_actions_per_replan':25,'mean_environment_actions':total/n*25,'max_mpc_replans':6,'physical_budget':150,'checkpoint_epoch':3,'checkpoint_sha256':checkpoint['sha256'],'raw_episode_files_checked':len(source['raw_episode_files_available']),'summary_only_conditions':source['summary_only_conditions']}
    assert abs(result['shift_rate']/100-summary['by_planner']['cem']['six_shift_success_rate'])<1e-10
    assert abs(result['color_rate']/100-summary['by_planner']['cem']['color_success_rate'])<1e-10
    return result

if __name__=='__main__':
    directory=Path(__file__).resolve().parents[1]/'data/dinowm_pusht_cem_epoch3'
    result=verified_pusht_cem(directory)
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
