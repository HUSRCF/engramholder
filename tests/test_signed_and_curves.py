"""Pairing and intervention-identity safeguards for the follow-up verifier."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'verify_signed_and_curves', Path(__file__).resolve().parents[1] / 'scripts/verify_signed_and_curves.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SignedAndCurvesTests(unittest.TestCase):
    def test_record_identity_rejects_duplicate_and_incomplete_grid(self):
        rows = [dict(model='F', target='a', status='ok'),
                dict(model='F', target='b', status='ok')]
        expected = {('F', 'a'), ('F', 'b')}
        self.assertEqual(len(module.index_records(rows[::-1], ('model', 'target'), expected)), 2)
        for changed in (rows + [rows[0]], rows[:-1],
                        rows + [dict(model='G', target='a', status='ok')]):
            with self.assertRaises(AssertionError):
                module.index_records(changed, ('model', 'target'), expected)

    def test_target_bootstrap_and_distinct_seed_transform_marginals(self):
        # Unequal axis sizes make accidental seed/transform transposition visible.
        delta = np.broadcast_to(np.array([-.1, .3]), (2, 3, 2)).copy()
        delta += np.array([-.02, .02])[:, None, None]
        delta += np.array([-.03, 0, .03])[None, :, None]
        draw = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        result = module.summarize(delta, draw)
        np.testing.assert_allclose(result['per_target'], [-.1, .3])
        np.testing.assert_allclose(result['per_seed'], [.07, .1, .13])
        np.testing.assert_allclose(result['per_rotation'], [.08, .12])
        np.testing.assert_allclose(result['ci95'], [-.085, .285])
        self.assertEqual(result['positive_targets'], 1)

    def test_signed_permutation_is_not_a_mean_preserving_rotation(self):
        spec = dict(convention='(T x)[j] = signs[j] * x[permutation[j]]',
                    permutation=[1, 2, 0], signs=[1, -1, 1])
        result = module.signed_transform_properties(spec)
        self.assertEqual(result['negative_signs'], 1)
        self.assertFalse(result['preserves_all_ones'])
        self.assertTrue(module.signed_transform_properties(dict(spec, signs=[1, 1, 1]))['preserves_all_ones'])
        with self.assertRaises(AssertionError):
            module.signed_transform_properties(dict(spec, permutation=[1, 1, 0]))


if __name__ == '__main__':
    unittest.main()
