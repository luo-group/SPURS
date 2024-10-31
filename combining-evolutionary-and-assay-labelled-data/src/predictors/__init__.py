from predictors import base_predictors, ev_predictors, onehot_predictors
from predictors import unirep_predictors, esm_predictors, vae_predictors
from predictors.base_predictors import BoostingPredictor, JointPredictor


BASIC_PREDICTOR_MAP = {
    'mutation': base_predictors.MutationRadiusPredictor,
    'ev': ev_predictors.EVPredictor,
    'onehot': onehot_predictors.OnehotRidgePredictor,
    'eunirep_reg': unirep_predictors.EUniRepRegressionPredictor,
    'gunirep_reg': unirep_predictors.GUniRepRegressionPredictor,
    'eunirep_ll': unirep_predictors.EUniRepLLPredictor,
    'gunirep_ll': unirep_predictors.GUniRepLLPredictor,
    'gesm': esm_predictors.GlobalESMPredictor,
    'gesm_reg': esm_predictors.GlobalESMRegressionPredictor,
    'vae': vae_predictors.VaePredictor,
    'ddg': esm_predictors.DDGPredictor,
}

CORE_PREDICTORS = [
    # 'onehot',
    # 'ev+onehot', 
    # 'gesm+onehot', 
    # 'eunirep_ll+onehot',
    'vae+onehot', 
    # 'ev+onehot+ddg', 
    # 'gesm+onehot+ddg', 
    # 'eunirep_ll+onehot+ddg',
    'vae+onehot+ddg'
]

OLD_CORE_PREDICTORS = [
    'eunirep_reg', 'ev+onehot', 'gesm+onehot', 'eunirep_ll+onehot',
    'vae+onehot',
]

BASELINE_PREDICTORS = [
    'georgiev', 'onehot', 'hmm+onehot', 'blosum+onehot', 'mutation+onehot',
]

ADDITIONAL_PREDICTORS = [
    'gunirep_ll+onehot',
    'gesm_reg',
]

UNSUPERVISED_PREDICTORS = [
    'ev', 'vae', 'hmm', 'blosum', 'mutation', 'eunirep_ll', 'gunirep_ll', 'gesm', 
]

def get_predictor_cls(predictor_name):
    names = predictor_name.split('+')
    return [BASIC_PREDICTOR_MAP[n] for n in names]

def get_predictor_names(key):
    if key == 'core':
        return CORE_PREDICTORS
    elif key == 'baselines':
        return BASELINE_PREDICTORS
    elif key == 'additional':
        return ADDITIONAL_PREDICTORS
    elif key == 'unsupervised':
        return UNSUPERVISED_PREDICTORS
    elif key == 'all':
        return CORE_PREDICTORS + BASELINE_PREDICTORS + ADDITIONAL_PREDICTORS + UNSUPERVISED_PREDICTORS
    else:
        return [key]
