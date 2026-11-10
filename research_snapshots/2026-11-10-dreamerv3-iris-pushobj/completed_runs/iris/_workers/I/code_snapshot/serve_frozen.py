"""Inference-only server using existing scorers; audit weights without training."""
import argparse
import hashlib
import os
from multiprocessing.connection import Listener
from pathlib import Path
import traceback

import numpy as np


def fingerprint(scorer,method):
    h=hashlib.sha256()
    if method=='dreamerv3':
        for key,value in sorted(scorer.parameters.items()):
            a=np.asarray(value);h.update(key.encode());h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    else:
        for name,module in [('tokenizer',scorer.tokenizer),('world_model',scorer.model)]:
            for key,value in module.state_dict().items():
                a=value.detach().cpu().contiguous().numpy();h.update((name+'.'+key).encode());h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('method',choices=['dreamerv3','iris'])
    ap.add_argument('--checkpoint-dir',type=Path,required=True);ap.add_argument('--socket',type=Path,required=True)
    ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:
        print('Dry run only; no model loaded.');return
    if a.method=='dreamerv3':
        from dreamer_model import DreamerScorer
        scorer=DreamerScorer(a.checkpoint_dir/'model_latest.pkl')
    else:
        from iris_model import IrisScorer
        scorer=IrisScorer(a.checkpoint_dir)
        scorer.tokenizer.requires_grad_(False);scorer.model.requires_grad_(False)
    if a.socket.exists():raise FileExistsError(a.socket)
    listener=Listener(str(a.socket),family='AF_UNIX');os.chmod(a.socket,0o600)
    try:
        while True:
            connection=listener.accept()
            try:
                while True:
                    request=connection.recv();op=request['operation']
                    try:
                        if op=='prepare':scorer.prepare(request);response={'ok':True}
                        elif op=='score':
                            costs=np.asarray(scorer.score(request['actions']),np.float32)
                            assert costs.shape==(len(request['actions']),) and np.isfinite(costs).all()
                            response={'losses':costs}
                        elif op=='fingerprint':response={'sha256':fingerprint(scorer,a.method)}
                        elif op=='ping':response={'ok':True}
                        elif op=='shutdown':connection.send({'ok':True});return
                        else:raise ValueError(op)
                    except Exception:
                        connection.send({'error':traceback.format_exc()});raise
                    connection.send(response)
            except EOFError:pass
            finally:connection.close()
    finally:
        listener.close()
        if a.socket.exists():a.socket.unlink()


if __name__=='__main__':main()
