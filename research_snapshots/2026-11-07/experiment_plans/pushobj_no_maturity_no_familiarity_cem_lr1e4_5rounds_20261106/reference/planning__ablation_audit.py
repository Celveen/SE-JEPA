"""Deterministic persistent-state audit; extracted unchanged from the experiment helper."""
import hashlib

import torch


def tree_hash(obj):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().contiguous()
            h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda y:str(y)):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(len(x)).encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(obj);return h.hexdigest()
