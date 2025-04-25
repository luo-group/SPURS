from sklearn.linear_model import Ridge, Lasso

from utils import seqs_to_onehot, seqs_to_georgiev
from predictors.base_predictors import BaseRegressionPredictor


class OnehotRidgePredictor(BaseRegressionPredictor):
    """Simple one hot encoding + ridge regression."""

    def __init__(self, dataset_name, reg_coef=1.0, **kwargs):
        super(OnehotRidgePredictor, self).__init__(
                dataset_name, reg_coef, Ridge, **kwargs)

    def seq2feat(self, seqs):
        return seqs_to_onehot(seqs)


class OnehotLassoPredictor(BaseRegressionPredictor):
    """Simple one hot encoding + lasso regression."""

    def __init__(self, dataset_name, reg_coef=1.0, **kwargs):
        super(OnehotLassoPredictor, self).__init__(
                dataset_name, reg_coef, Lasso, **kwargs)

    def seq2feat(self, seqs):
        return seqs_to_onehot(seqs)


# class OnehotGPPredictor(BaseGPPredictor):

#     def seq2feat(self, seqs):
#         return seqs_to_onehot(seqs)


class GeorgievRidgePredictor(BaseRegressionPredictor):
    """Georgiev encoding + ridge regression."""

    def __init__(self, dataset_name, reg_coef=1.0, **kwargs):
        super(GeorgievRidgePredictor, self).__init__(
                dataset_name, reg_coef, Ridge, **kwargs)

    def seq2feat(self, seqs):
        return seqs_to_georgiev(seqs)
