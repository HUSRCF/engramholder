"""Guard the common-control pairing and whole-target bootstrap unit in E2."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'verify_e2_intervention', Path(__file__).resolve().parents[1] / 'scripts/verify_e2_intervention.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PairedInterventionTest(unittest.TestCase):
    def test_common_native_is_subtracted_once_after_rotation_averaging(self):
        native = np.zeros((1, 2))
        native_c = np.array([[0., 1.]])
        rotated = np.zeros((2, 1, 2))
        rotated_c = np.array([[[0., 2.]], [[2., 0.]]])
        result = module.paired_effects(native, native_c, rotated, rotated_c)
        np.testing.assert_allclose(result['B_R'], [[1., 1.]])
        np.testing.assert_allclose(result['T_C'], [[1., 0.]])
        # Four equally likely resamples of two target records. The two rotation
        # records for a target remain together; they are not four independent units.
        draws = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        summary = module.conditional_summary(result['T_C'], draws)
        self.assertEqual(summary['mean'], .5)
        np.testing.assert_allclose(summary['ci95'], [.0375, .9625])

    def test_mean_gap_reduction_does_not_imply_rotated_improvement(self):
        native = np.ones((2, 3))
        native_c = native - .2
        rotated = np.full((2, 2, 3), .8)
        result = module.paired_effects(native, native_c, rotated, rotated)
        np.testing.assert_allclose(result['T_C'], .2)
        np.testing.assert_allclose(result['B_R'], 0.)
        np.testing.assert_allclose(result['native_change'], -.2)

    def test_unpaired_or_invalid_arrays_are_rejected(self):
        native = np.zeros((3, 4))
        rotated = np.zeros((2, 3, 4))
        with self.assertRaises(ValueError):
            module.paired_effects(native, native, rotated, rotated[:, :, :-1])
        with self.assertRaises(ValueError):
            module.paired_effects(native, native, rotated, rotated + np.nan)
        with self.assertRaises(ValueError):
            module.conditional_summary(native, np.zeros((2, 3), dtype=int))
        with self.assertRaises(ValueError):
            module.conditional_summary(native, np.full((2, 4), 4, dtype=int))


if __name__ == '__main__':
    unittest.main()
