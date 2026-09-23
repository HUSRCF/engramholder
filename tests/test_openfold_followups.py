"""Pairing safeguards for the new score-only analyses."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('followups',Path(__file__).resolve().parents[1]/'scripts/analyze_openfold_followups.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class PairingTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for tid,offset in [('x',0),('y',.1)]:
            for s in m.SEEDS:
                for name,val in [(f'native_s{s}',.7),(f'gplus_s{s}',.6)]:
                    rows.append(dict(panel='p',system=name,target_id=tid,status='ok',score=val+offset))
                for r in m.ROTATIONS:
                    for name,val in [(f'r{r}_s{s}',.4),(f'gplus_r{r}_s{s}',.5)]:
                        rows.append(dict(panel='p',system=name,target_id=tid,status='ok',score=val+offset))
        return rows

    def test_known_interaction_and_target_pairing(self):
        a=m.arrays(self.rows(),'p',['x','y'],'score')
        np.testing.assert_allclose(a['psi'],.2)
        b=m.arrays(self.rows()[::-1],'p',['y','x'],'score')
        np.testing.assert_allclose(a['psi'],b['psi'][:,:,::-1])

    def test_duplicate_cannot_silently_replace_score(self):
        rs=self.rows()
        with self.assertRaisesRegex(AssertionError,'Duplicate'):
            m.arrays(rs+[rs[0]],'p',['x','y'],'score')

    def test_bootstrap_keeps_model_cells_together(self):
        x=np.broadcast_to(np.array([-.1,.3]),(3,3,2))
        draw=np.array([[0,0],[1,1],[0,1],[1,0]])
        result=m.summarize(x,['x','y'],draw)
        self.assertEqual(result['positive_targets'],1)
        np.testing.assert_allclose(result['per_target'],[-.1,.3])
        np.testing.assert_allclose(result['ci95'],[-.085,.285])


if __name__=='__main__':unittest.main()
