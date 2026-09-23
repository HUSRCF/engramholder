"""Primary-score conventions and CIF residue mapping in the portable example."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

path=Path(__file__).resolve().parents[1]/'reproducibility/openfold_single/score.py'
spec=importlib.util.spec_from_file_location('single_score',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ScoreTests(unittest.TestCase):
    def test_threshold_is_strict_and_missing_prediction_stays_in_denominator(self):
        ref={1:np.array([0.,0.,0.]),2:np.array([4.,0.,0.])}
        shifted={1:ref[1],2:np.array([4.5,0.,0.])}
        self.assertEqual(m.score(ref,shifted)['ca_pair_lddt'],.75)
        self.assertEqual(m.score(ref,{1:ref[1]})['ca_pair_lddt'],0)
        self.assertEqual(m.score(ref,{1:ref[1]})['reference_pair_count'],1)

    def test_reference_distance_cutoff_is_strict(self):
        ref={1:np.array([0.,0.,0.]),2:np.array([1.,0.,0.]),3:np.array([15.,0.,0.])}
        result=m.score(ref,ref)
        self.assertEqual(result['reference_pair_count'],2)
        self.assertEqual(result['ca_pair_lddt'],1.)

    def test_label_indices_and_altloc_priority(self):
        import gemmi
        doc=gemmi.cif.Document();b=doc.add_new_block('test')
        loop=b.init_loop('_atom_site.', ['group_PDB','label_asym_id','label_seq_id','label_atom_id',
            'Cartn_x','Cartn_y','Cartn_z','pdbx_PDB_model_num','label_alt_id'])
        for chain,index,x,alt in [('A','2','99','B'),('A','2','4','A'),('B','1','11','.')]:
            loop.add_row(['ATOM',chain,index,'CA',x,'0','0','1',alt])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.cif';doc.write_file(str(p));result=m.ca_coordinates(p,'A')
        self.assertEqual(set(result),{2});np.testing.assert_equal(result[2],[4,0,0])


if __name__=='__main__':unittest.main()
