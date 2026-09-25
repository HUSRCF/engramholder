"""Guard statistical units, complete pairing and the locked failure denominator."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_protenix_fresh192 import BINS, METRICS, effects, score_grid, stratified_draws, system_names


class Fresh192Test(unittest.TestCase):
    def test_stratified_draws_keep_all_four_quotas(self):
        targets = [{'length_bin': b} for b in BINS for _ in range(48)]
        draw = stratified_draws(targets, draws=20)
        self.assertEqual(draw.shape, (20, 192))
        for i in range(4):
            np.testing.assert_array_equal(((draw >= i*48) & (draw < (i+1)*48)).sum(1), 48)
        with self.assertRaises(ValueError):
            stratified_draws(targets[:-1])

    def test_interaction_requires_both_rows_and_keeps_target_pairing(self):
        n = np.array([[.7, .5], [.8, .6]])
        g = n-.1
        r = np.stack([n-.02, n-.04])
        gr = np.stack([g-.01, g+.01])
        result = effects(n, r, g, gr, np.array([.3, .4]))
        np.testing.assert_allclose(result['interaction'].mean((0, 1)), [.03, .03])
        np.testing.assert_allclose(result['interaction'], result['factor_rotation']-result['gplus_rotation'])
        self.assertEqual(result['native_minus_query'].shape, (1, 2, 2))

    def test_duplicate_or_missing_grid_is_rejected(self):
        rows = [dict(system=s, target_id='one', status='ok', **dict.fromkeys(METRICS, .5)) for s in system_names()]
        score_grid(list(reversed(rows)), ['one'])
        for bad in [rows[:-1], rows+[rows[0]], rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):
                score_grid(bad, ['one'])

    def test_failed_prediction_must_remain_zero_in_complete_grid(self):
        rows = [dict(system=s, target_id='one', status='ok', **dict.fromkeys(METRICS, .5)) for s in system_names()]
        failed = copy.deepcopy(rows)
        failed[0].update(status='failed', **dict.fromkeys(METRICS, 0.))
        self.assertEqual(len(score_grid(failed, ['one'])), 25)
        failed[0]['ca_lddt'] = .1
        with self.assertRaises(ValueError):
            score_grid(failed, ['one'])


if __name__ == '__main__':
    unittest.main()
