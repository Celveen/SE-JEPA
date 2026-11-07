"""CPU-only tests of removing familiarity filtering while keeping relative selection."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]/'CL-JEPA-cudagraph'
sys.path.insert(0,str(ROOT))
from planning.continual import ContinualMPCPlanner
from planning.continual_v7 import CLSv7MPCPlanner


def make(disabled):
    p=CLSv7MPCPlanner.__new__(CLSv7MPCPlanner)
    p.familiar_thr=0. if disabled else .117;p.familiar_read=not disabled
    p.mature_k=3;p.mature_ratio=.9
    p.domains={'d0':{'learner':'learner','val':[.4,.5,.6]}}
    p._obs_buffer=[];p._act_buffer=[];p.iter=0
    p._extract_adajepa_data=lambda *args:(None,None)
    p._theta0_latents=lambda *args:None;p._familiarity=lambda z:.001
    p._pred0_module='base';p._latent_loss=lambda module,z:.01 if module=='base' else .005
    p._base=None;p._route_log=[]
    p._load_base=lambda key,carry_fast:setattr(p,'_base',key)
    return p


class FamiliarityRemovalTest(unittest.TestCase):
    def test_relative_error_can_choose_learner_on_previously_familiar_input(self):
        old=make(False);new=make(True)
        old._route(None,None);new._route(None,None)
        self.assertIsNone(old._base);self.assertEqual(new._base,'d0')
        self.assertEqual(set(new._route_log[-1]['errs']),{'None','d0'})

    def test_base_still_wins_when_it_has_lower_error(self):
        p=make(True);p._latent_loss=lambda module,z:.001 if module=='base' else .005
        p._route(None,None);self.assertIsNone(p._base)

    def test_maturity_filter_remains(self):
        p=make(True);p.domains['d0']['val']=[]
        p._route(None,None);self.assertIsNone(p._base)
        self.assertEqual(set(p._route_log[-1]['errs']),{'None'})

    def test_familiarity_no_longer_skips_write_routing(self):
        def record(p):
            p._ep_records.append({'episode':p._episode});p._episode+=1
        results=[]
        for disabled in (False,True):
            p=make(disabled);p.read_only=False;p._params=[];p._start=[];p.domains={}
            p._episode_latents=lambda:None;p._rollout_err=lambda *args:1.
            p.join_ratio=0;p._abs_join=lambda *args:None
            p.pending=[];p._ep_records=[];p._episode=0;p.absorb_every=0;p.pend_calls=0
            def pend(*args):
                p.pend_calls+=1
                return 'pending',None,[]
            p._pend=pend
            with patch.object(ContinualMPCPlanner,'_end_episode',record):p._end_episode()
            results.append((p.pend_calls,p._ep_records[-1]['write_rule']))
        self.assertEqual(results,[(0,'familiar'),(1,'pending')])


if __name__=='__main__':unittest.main()
