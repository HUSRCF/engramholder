"""Same native predictor; only add the separately locked eight-target Dev role."""
from .common import manifest_validate
from engramfold.experiments.protenix_fresh_fourcell import predict
if __name__=='__main__':
 predict.validate_manifest=manifest_validate
 predict.main()
