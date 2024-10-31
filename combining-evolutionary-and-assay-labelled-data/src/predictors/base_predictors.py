import os
import numpy as np

from Bio.Align import substitution_matrices
from sklearn.linear_model import LinearRegression, Ridge
# from skopt.learning import GaussianProcessRegressor
# from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import utils
import glob
from utils import seqs_to_onehot, read_fasta, load_rows_by_numbers
import pandas as pd
import pickle

REG_COEF_LIST = [1e-1, 1e0, 1e1, 1e2, 1e3]


class BasePredictor():
    """Abstract class for predictors."""

    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name

    def select_training_data(self, data, n_train):
        return data.sample(n=n_train)

    def train(self, train_seqs, train_labels):
        """Trains the model.
        Args:
            - train_seqs: a list of sequences
            - train_labels: a list of numerical fitness labels
        """
        raise NotImplementedError

    def predict(self, predict_seqs):
        """Gets model predictions.
        Args:
            - predict_seqs: a list of sequences
        Returns:
            A list of numerical fitness predictions.
        """
        raise NotImplementedError

    def predict_unsupervised(self, predict_seqs):
        """Gets model predictions before training.
        Args:
            - predict_seqs: a list of sequences
        Returns:
            A list of numerical fitness predictions.
        """
        return np.random.randn(len(predict_seqs))


class BoostingPredictor(BasePredictor):
    """Boosting by combining predictors as weak learners."""

    def  __init__(self, dataset_name, weak_learner_classes, **kwargs):
        super(BoostingPredictor, self).__init__(dataset_name)
        self.weak_learners = [c(dataset_name, **kwargs) for c in
                weak_learner_classes]

    def train(self, train_seqs, train_labels):
        y = train_labels
        for i, model in enumerate(self.weak_learners):
            model.train(train_seqs, y)
            y -= model.predict(train_seqs)

    def predict(self, predict_seqs):
        y = np.zeros(len(predict_seqs))
        for i, model in enumerate(self.weak_learners):
            y += model.predict(predict_seqs)
        return y

    def predict_unsupervised(self, predict_seqs):
        return self.weak_learners[0].predict_unsupervised(predict_seqs)

    def select_training_data(self, data, n_train):
        return self.weak_learners[0].select_training_data(data, n_train)

class DDGPredictor_null():
    def __init__(self, dataset_name, reg_coef=1e-8, path_prefix='',
            **kwargs):
        super(DDGPredictor_null, self).__init__()
        fast_eval_files = glob.glob(f'{os.getenv("PROJECT_ROOT", default=os.getcwd())}/data/fitness/proteingm_groundtruth/*/')
        fast_eval = dataset_name in [i.split('/')[-2] for i in fast_eval_files]
        seqs_path = path_prefix + os.path.join('data', dataset_name, 'seqs.fasta') if not fast_eval else path_prefix + os.path.join(f'{os.getenv("PROJECT_ROOT", default=os.getcwd())}/data/fitness/proteingm_groundtruth', dataset_name, 'seqs.fasta')
        seqs = read_fasta(seqs_path)
        id2seq = pd.Series(index=np.arange(len(seqs)), data=seqs, name='seq')

        esm_data_path = path_prefix + os.path.join('inference', dataset_name,
                'esm', 'pll.csv') if not fast_eval else path_prefix + os.path.join(f'{os.getenv("PROJECT_ROOT", default=os.getcwd())}/data/fitness/proteingym_deepsequence', dataset_name, 'DeepSequence', 'pll.csv')
        spurs_data_path = path_prefix + os.path.join('inference', dataset_name, 'spurs.pkl') if not fast_eval else path_prefix + os.path.join(f'{os.getenv("PROJECT_ROOT", default=os.getcwd())}/data/fitness/proteingym_deepsequence', dataset_name, 'spurs.pkl')

        with open(spurs_data_path, 'rb') as f:
            data = pickle.load(f)
        ll = pd.read_csv(esm_data_path, index_col=0)
        ll['id'] = ll.index.to_series().apply(
                lambda x: int(x.replace('id_', '')))
        
        ll = ll.join(id2seq, on='id', how='left')
        ll['ddg'] = data
        self.seq2score_dict = dict(zip(ll.seq, ll.ddg))


    def seq2score(self, seqs):
        scores = np.array([self.seq2score_dict.get(s, 0.0) for s in seqs])
        return scores

    def seq2feat(self, seqs):
        return self.seq2score(seqs)[:, None].reshape(len(seqs),-1)

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs) 
class BaseRegressionPredictor(BasePredictor):

    def __init__(self, dataset_name, reg_coef=None,
            linear_model_cls=Ridge, **kwargs):
        self.dataset_name = dataset_name
        self.reg_coef = reg_coef
        self.linear_model_cls = linear_model_cls
        self.model = None

        self.DDGPredictor = DDGPredictor_null(dataset_name)
    def seq2feat(self, seqs):
        raise NotImplementedError

    def train(self, train_seqs, train_labels):
        X = self.seq2feat(train_seqs)
        if self.reg_coef is None or self.reg_coef == 'CV':
            best_rc, best_score = None, -np.inf
            for rc in REG_COEF_LIST:
                model = self.linear_model_cls(alpha=rc)
                score = cross_val_score(
                    model, X, train_labels, cv=5,
                    scoring=make_scorer(utils.spearman)).mean()
                if score > best_score:
                    best_rc = rc
                    best_score = score
            self.reg_coef = best_rc
            # print(f'Cross validated reg coef {best_rc}')
        self.model = self.linear_model_cls(alpha=self.reg_coef)
        self.model.fit(X, train_labels)

    def predict(self, predict_seqs,ret_values=False):
        if self.model is None:
            return np.random.randn(len(predict_seqs))

        if ret_values:
            X = self.seq2feat(predict_seqs)
            if len(self.predictors)==3:
                vae_score = X[:,0]
                ddg_score = X[:,-1]
                return self.model.predict(X),vae_score/np.sqrt(1.0 / self.predictors[0].reg_coef),ddg_score/np.sqrt(1.0 / self.predictors[-1].reg_coef)
            elif len(self.predictors)==2:
                x = self.DDGPredictor.seq2feat(predict_seqs)
                
                vae_score = X[:,0]
                ddg_score = x.squeeze()
                return self.model.predict(X),vae_score/np.sqrt(1.0 / self.predictors[0].reg_coef),ddg_score
       
        X = self.seq2feat(predict_seqs)
        return self.model.predict(X)


class JointPredictor(BaseRegressionPredictor):
    """Combining regression predictors by training jointly."""

    def  __init__(self, dataset_name, predictor_classes, predictor_name,
        reg_coef='CV', **kwargs):
        super(JointPredictor, self).__init__(dataset_name, reg_coef, **kwargs)
        self.predictors = []
        for c, name in zip(predictor_classes, predictor_name):
            if f'reg_coef_{name}' in kwargs:
                self.predictors.append(c(dataset_name,
                    reg_coef=float(kwargs[f'reg_coef_{name}']), **kwargs))
            else:
                self.predictors.append(c(dataset_name, **kwargs))

        
    def seq2feat(self, seqs):
        # To apply different regularziation coefficients we scale the features
        # by a multiplier in Ridge regression
        features = [p.seq2feat(seqs) * np.sqrt(1.0 / p.reg_coef)
            for p in self.predictors]
        

        return np.concatenate(features, axis=1)


class BLOSUM62Predictor(BaseRegressionPredictor):

    def __init__(self, dataset_name, reg_coef=1e-8, **kwargs):
        super(BLOSUM62Predictor, self).__init__(dataset_name,
            reg_coef, **kwargs)
        self.wt = utils.read_fasta(
            os.path.join('data', dataset_name, 'wt.fasta'))[0]
        self.matrix = substitution_matrices.load('BLOSUM62')
        self.alphabet = self.matrix.alphabet
        for i, c in enumerate(self.wt):
            assert c in self.alphabet, f'unexpected AA {c} (pos {i})'

    def seq2feat(self, seqs):
        scores = np.zeros(len(seqs))
        return utils.get_blosum_scores(seqs, self.wt, self.matrix)[:, None]

    def predict_unsupervised(self, predict_seqs):
        return self.seq2feat(predict_seqs).squeeze()


# class BaseGPPredictor(BasePredictor):

#     def __init__(self, dataset_name, noise=0.1, kernel_length_scale=1.0,
#             kernel_nu=2.5, kernel_const=1.0, **kwargs):
#         self.dataset_name = dataset_name
#         self.kernel = ConstantKernel(kernel_const) * Matern(
#                 length_scale=kernel_length_scale, nu=kernel_nu)
#         self.noise = noise
#         self.gpr = None

#     def seq2feat(self, seqs):
#         raise NotImplementedError

#     def train(self, train_seqs, train_labels):
#         self.gpr = GaussianProcessRegressor(kernel=self.kernel,
#                 alpha=self.noise**2)
#         X = self.seq2feat(train_seqs)
#         # Use negative labels for minimization.
#         self.gpr = self.gpr.fit(X, -train_labels)

#     def predict(self, predict_seqs):
#         if self.gpr is None:
#             return np.random.randn(len(predict_seqs))
#         X = self.seq2feat(predict_seqs)
#         return -self.gpr.predict(X, return_std=False)


class RandomPredictor(BasePredictor):

    def train(self, train_seqs, train_labels):
        self.train_labels = train_labels

    def predict(self, predict_seqs):
        return np.random.choice(self.train_labels, size=len(predict_seqs),
                replace=True)


class MutationRadiusPredictor(BaseRegressionPredictor):

    def __init__(self, dataset_name, reg_coef=1e-8, **kwargs):
        super(MutationRadiusPredictor, self).__init__(dataset_name,
            reg_coef, **kwargs)
        self.wt = utils.read_fasta(os.path.join('data', dataset_name, 'wt.fasta'))[0]

    def seq2feat(self, seqs):
        mutation_counts = np.zeros(len(seqs))
        for i, s in enumerate(seqs):
            for j in range(len(self.wt)):
                if self.wt[j] != s[j]:
                    mutation_counts[i] += 1
        return -mutation_counts[:, None]

    def predict_unsupervised(self, predict_seqs):
        return self.seq2feat(predict_seqs).squeeze()


def select_training_data(data, n_train, scores):
    sorted_idx = np.argsort(scores)
    idx = sorted_idx[-n_train:]
    return data.iloc[idx, :].sample(n=n_train)
