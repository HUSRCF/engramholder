"""Preserve sealed source; narrow only the Dev metadata lookup to its sealed IDs."""
import importlib.util,json,os,traceback
from pathlib import Path
from engramfold.experiments.protenix_budget_extension import score
from engramfold.experiments.protenix_budget_extension.common import ROOT,read,write,sha,stamp
HERE=Path(__file__).resolve().parent

def main():
    amendment=read(HERE/'amendment.json')
    for path,digest in amendment['files'].items():
        assert sha(path)==digest,path
    assert sha(ROOT/'execution_lock.json')==amendment['execution_lock_sha256']
    assert sha(ROOT/'analysis/metric_records.json')==amendment['original_metric_records_sha256']
    spec=importlib.util.spec_from_file_location('budget_reference_fix',HERE/'reference_subset.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    manifest=read(ROOT/'inputs/dev.json');references=read(ROOT/'inputs/dev_reference.json')
    selected=helper.select_reference_targets(manifest,references)
    assert len(references['targets'])==392 and len(selected['targets'])==8
    # This reproduces the old failure without consulting any outcome or selecting a score.
    scored=read(ROOT/'analysis/metric_records.json')
    expected={(x['panel'],x['node'],x['system'],x['target_id']) for x in read(ROOT/'prediction_completion.json')['records']}
    actual=[(x['panel'],x['node'],x['system'],x['target_id']) for x in scored]
    assert len(expected)==len(actual)==len(set(actual))==9984 and set(actual)==expected
    reference_ids=[t['target_id'] for t in selected['targets']]
    for step in [1536,2304,3072]:
        ids={x['target_id'] for x in scored if x['panel']=='dev' and x['node']==step}
        assert ids==set(reference_ids)
    original_read=score.read
    def panel_bound_read(path):
        value=original_read(path)
        if Path(path).resolve()==(ROOT/'inputs/dev_reference.json').resolve():
            return helper.select_reference_targets(original_read(ROOT/'inputs/dev.json'),value)
        return value
    score.read=panel_bound_read
    original_write=score.write
    def audited_write(path,value):
        if Path(path) in [ROOT/'analysis/analysis.json',ROOT/'analysis/complete.json']:
            value=dict(value,recovery_amendment_sha256=sha(HERE/'amendment.json'))
        original_write(path,value)
    score.write=audited_write
    write(HERE/'preflight.json',dict(passed=True,original_reference_targets=392,actual_dev_targets=reference_ids,metric_records=9984,selection_uses_locked_ids_only=True,time=stamp()))
    write(ROOT/'status.json',dict(state='RUNNING',stage='scoring_reference_selection_recovery',time=stamp(),preserved_failure=str(HERE/'failed_status.json')))
    try:
        score.main()
        assert sha(ROOT/'analysis/metric_records.json')==amendment['original_metric_records_sha256'],'rescoring changed raw metrics'
        write(HERE/'complete.json',dict(complete=True,amendment_sha256=sha(HERE/'amendment.json'),all_9984_metric_records_identical=True,analysis_sha256=sha(ROOT/'analysis/analysis.json'),time=stamp()))
        write(ROOT/'status.json',dict(state='COMPLETE',stage='COMPLETE',time=stamp(),scoring_recovery=str(HERE/'complete.json'),original_scoring_failure_preserved=True))
    except BaseException:
        write(ROOT/'status.json',dict(state='FAILED',stage='scoring_reference_selection_recovery',time=stamp(),error=traceback.format_exc()));raise
if __name__=='__main__':main()
