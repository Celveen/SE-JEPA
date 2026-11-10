"""Extract recorded episode observations; no simulation or image alteration.

Usage: python tools/extract_mixed14_observations.py <mixed14 raw archive root>
The torch ground-truth action tensor is deliberately skipped; image/state arrays
are original NumPy values. Only the supplied experiment archive is supported.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, pickle
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
DOMAINS=[('T','T'),('L','L'),('Z','Z'),('plus','Plus'),('I','I'),('square','Square'),('small_tee','Small T'),('default','Original'),('blur','Blur'),('snp1','Noise'),('dark','Dark'),('red_agent','Red agent'),('red_object','Red object'),('red_goal','Red goal')]
class ImagesOnly(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='torch.storage' and name=='_load_from_bytes':
            return lambda blob: None
        if module=='torch._utils' and name in ['_rebuild_tensor_v2','_rebuild_tensor']:
            return lambda *args: None
        return super().find_class(module,name)
def main():
    cli=argparse.ArgumentParser();cli.add_argument('archive',type=Path);args=cli.parse_args()
    dest=ROOT/'data/revision_assets/mixed14_observations';dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for domain,label in DOMAINS:
        runs=list((args.archive/'results/GD/v6a/repeat_2').glob(f'chunk_*_r5_{domain}'));assert len(runs)==1
        run=runs[0];source=run/'plan_targets.pkl.gz'
        with gzip.open(source,'rb') as f:payload=f.read()
        receipt=json.loads((run/'target_archive.json').read_text())
        assert hashlib.sha256(payload).hexdigest()==receipt['sha256']
        import io
        data=ImagesOnly(io.BytesIO(payload)).load()
        with np.load(run/'final_eval_results.npz') as results:
            assert np.array_equal(data['state_0'],results['state_0'])
            assert np.array_equal(data['state_g'],results['state_g'])
            success=bool(results['successes'][0])
        frame=data['obs_0']['visual'][0,0]
        assert frame.shape==(224,224,3) and frame.dtype==np.uint8
        output=dest/f'{domain}.png';Image.fromarray(frame).save(output)
        assert np.array_equal(frame,np.asarray(Image.open(output)))
        records.append({'domain':domain,'label':label,'family':'shape' if len(records)<7 else 'visual','image':str(output.relative_to(ROOT)),'sample_index':0,'planner':'GD','chain':2,'round':5,'frame':'recorded initial observation','source':str(source.relative_to(args.archive)),'source_archive_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_pickle_sha256':receipt['sha256'],'image_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_results_sha256':hashlib.sha256((run/'final_eval_results.npz').read_bytes()).hexdigest(),'episode_success':success,'state_0':data['state_0'][0].tolist(),'state_g':data['state_g'][0].tolist()})
    audit={'selection':'First episode (index 0) of every domain in GD chain 2, round 5; fixed index, independent of outcome.','display':'One original 224x224 simulator observation per domain, including actual recorded visual corruption; no regeneration or pixel editing.','role':'Illustration of stream domains; not a success/failure comparison or additional performance evidence.','archive_root_name':args.archive.name,'cases':records}
    (dest/'manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(f'Extracted {len(records)} pixel-exact recorded observations.')
if __name__=='__main__':main()
