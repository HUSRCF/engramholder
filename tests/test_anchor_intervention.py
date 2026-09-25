"""Guard direction, common-control pairing and the actual archived anchor hook."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from verify_anchor_intervention import paired_effects

ARCHIVE = ROOT/'reproducibility/anchor_intervention/archived'
sys.path.insert(0, str(ARCHIVE))
spec = importlib.util.spec_from_file_location('archived_anchor_tests', ARCHIVE/'test_query_anchor.py')
hook_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook_tests)


class AnchorInterventionTest(unittest.TestCase):
    def test_live_rotation_benefit_and_shared_native_counted_once(self):
        nl = np.full((2, 3), .6)
        nq = nl + .01
        rl = np.full((2, 2, 3), .5)
        rq = rl - np.array([.02, .04])[:, None, None]
        effect = paired_effects(nl, nq, rl, rq)
        np.testing.assert_allclose(effect['B_R_live_minus_Q'], .03)
        np.testing.assert_allclose(effect['T_A'], .04)
        np.testing.assert_allclose(effect['rotation_T_A'],
                                   effect['rotation_B_R_live_minus_Q']+.01)

    def test_rotated_improvement_can_be_canceled_by_native_change(self):
        nl = np.full((1, 2), .6)
        rl = np.full((2, 1, 2), .5)
        effect = paired_effects(nl, nl-.02, rl, rl-.02)
        np.testing.assert_allclose(effect['B_R_live_minus_Q'], .02)
        np.testing.assert_allclose(effect['T_A'], 0., atol=1e-15)

    def test_invalid_pairing_or_nonfinite_scores_are_rejected(self):
        n, r = np.zeros((3, 4)), np.zeros((2, 3, 4))
        with self.assertRaises(ValueError):
            paired_effects(n, n, r, r[:, :, :-1])
        with self.assertRaises(ValueError):
            paired_effects(n, n, r, r+np.nan)

    def test_archived_hook_preserves_baseline_and_pins_backward_round(self):
        hook_tests.test_reference_only_changes_residual_and_backward_reuses_last_anchor()

    def test_archived_hook_rejects_reference_mask_mismatch(self):
        module, writer, _, refs = hook_tests.setup()
        hook = hook_tests.QueryAnchorHook(module, writer, torch.zeros(2, 1), refs)
        try:
            with self.assertRaisesRegex(ValueError, 'mask/normalization'):
                module(torch.ones(1, 2, 1), torch.tensor([[1., 0.]]))
        finally:
            hook.remove()


if __name__ == '__main__':
    unittest.main()
