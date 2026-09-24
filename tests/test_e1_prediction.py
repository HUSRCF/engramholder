"""Check the independent exact rank test's tail, ties, and undefined cases."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'verify_e1_prediction', Path(__file__).resolve().parents[1] / 'scripts/verify_e1_prediction.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExactPredictionTest(unittest.TestCase):
    def test_eight_locked_rank_orders(self):
        y = [6, 4, 8, 3, 1, 5, 7, 2]
        x = module.exact_positive_rank_test([6, 5, 7, 8, 4, 2, 1, 3], y)
        d = module.exact_positive_rank_test([8, 6, 7, 4, 3, 2, 1, 5], y)
        self.assertEqual(x['extreme_count'], 19692)
        self.assertEqual(d['extreme_count'], 13398)
        self.assertEqual(x['permutations'], 40320)
        self.assertAlmostEqual(x['rho'], 1/42)
        self.assertAlmostEqual(d['rho'], 4/21)

    def test_positive_tail_includes_equal_statistics_and_labelled_ties(self):
        positive = module.exact_positive_rank_test([1, 2, 3], [1, 2, 3])
        negative = module.exact_positive_rank_test([1, 2, 3], [3, 2, 1])
        tied = module.exact_positive_rank_test([1, 1, 2], [1, 1, 2])
        self.assertEqual(positive['p_one_sided'], 1/6)
        self.assertEqual(negative['p_one_sided'], 1)
        self.assertEqual(tied['p_one_sided'], 2/6)

    def test_invalid_unit_arrays_are_rejected(self):
        for x, y in [([1, 1], [1, 2]), ([1, 2], [1, 2, 3]),
                     ([1, float('nan')], [1, 2]), ([1], [2])]:
            with self.assertRaises(ValueError):
                module.exact_positive_rank_test(x, y)


if __name__ == '__main__':
    unittest.main()
