"""Join reference metadata to the sealed sequence panel, never infer membership."""
def select_reference_targets(manifest, references):
    targets = manifest['targets']
    ids = [t['target_id'] for t in targets]
    if len(ids) != 8 or len(set(ids)) != 8 or manifest.get('role') != 'budget_development':
        raise ValueError('requires the locked eight-target development sequence panel')
    rows = references['targets']
    by_id = {t['target_id']: t for t in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate reference target IDs')
    selected = []
    for target in targets:
        tid = target['target_id']
        if tid not in by_id:
            raise ValueError('missing development reference: ' + tid)
        reference = by_id[tid]
        for key in ('sequence', 'sequence_sha256', 'sequence_length'):
            if reference[key] != target[key]:
                raise ValueError('development sequence/reference mismatch: ' + tid + '/' + key)
        selected.append(reference)
    return dict(references, targets=selected)
