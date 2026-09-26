"""Matrix and selection rules for the prospectively locked schedule retraining."""
SEEDS = (20260923, 20260924, 20260925)
ROTATIONS = (20261001, 20261002, 20261003)
STRATEGIES = ('first', 'all', 'last')
LEARNING_RATES = (5e-5, 1e-4)


def formal_matrix():
    rows = []
    for wave, rotations in ((1, (None, ROTATIONS[0])), (2, ROTATIONS[1:])):
        for strategy in STRATEGIES:
            for seed in SEEDS:
                for kind in ('factor', 'generic_plus'):
                    for rotation in rotations:
                        name = f'{strategy}_{kind}_{"native" if rotation is None else "r"+str(rotation)}_s{seed}'
                        rows.append(dict(name=name, strategy=strategy, seed=seed,
                                         kind=kind, rotation=rotation, steps=1536, wave=wave))
    assert len(rows) == len({r['name'] for r in rows}) == 72
    return rows


def calibration_matrix():
    # A bounded common recipe search uses the predesignated first rotation only.
    return [dict(name=f'cal_{s}_{k}_{rot}_lr{lr:g}', strategy=s, kind=k,
                 rotation=rot, seed=20260922, steps=192, lr=lr)
            for lr in LEARNING_RATES for s in STRATEGIES
            for k in ('factor', 'generic_plus') for rot in (None, ROTATIONS[0])]


def choose_lr(reports, dev_ids):
    import math
    expected = calibration_matrix()
    assert set(reports) == {r['name'] for r in expected}
    scores = {}
    for lr in LEARNING_RATES:
        values = []
        for r in expected:
            if r['lr'] != lr:
                continue
            report = reports[r['name']]
            assert report['complete'] and report['failures'] == 0
            rows = report['scores']
            assert [x['target_id'] for x in rows] == list(dev_ids)
            values.extend(x['ca_lddt'] for x in rows)
        assert len(values) == 12 * len(dev_ids) and all(math.isfinite(v) for v in values)
        scores[lr] = sum(values) / len(values)
    return dict(learning_rate=max(LEARNING_RATES, key=lambda lr: (scores[lr], -lr)),
                scores=scores, rule='mean absolute Dev8 pair-lDDT over all 12 strategy/arm cells; exact ties choose smaller LR')
