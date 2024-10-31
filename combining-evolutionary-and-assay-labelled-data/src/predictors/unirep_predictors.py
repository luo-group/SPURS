import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from utils import load, load_rows_by_numbers
from utils import seqs_to_onehot, get_wt_seq, read_fasta, seq2effect
from predictors.base_predictors import BaseRegressionPredictor


class BaseUniRepPredictor(BaseRegressionPredictor):
    """UniRep representation + regression."""

    def __init__(self, dataset_name, rep_name, reg_coef=1.0, **kwargs):
        super(BaseUniRepPredictor, self).__init__(
            dataset_name, reg_coef, Ridge)
        
        # self.load_rep(dataset_name, rep_name)

    def load_rep(self, dataset_name, rep_name):
        self.rep_path = os.path.join('inference', dataset_name,
                'unirep', rep_name, f'avg_hidden.npy*')
        self.seq_path = os.path.join('inference', dataset_name,
                'unirep', rep_name, f'seqs.npy')
        #self.features = load(self.rep_path)
        self.seqs = np.loadtxt(self.seq_path, dtype=str, delimiter=' ') 
        self.seq2id = dict(zip(self.seqs, range(len(self.seqs))))

    def seq2feat(self, seqs):
        """Look up representation by sequence."""
        ids = [self.seq2id[s] for s in seqs]
        return load_rows_by_numbers(self.rep_path, ids)


class GUniRepRegressionPredictor(BaseUniRepPredictor):
    """Global UniRep + Ridge regression."""

    def __init__(self, dataset_name, **kwargs):
        super(GUniRepRegressionPredictor, self).__init__(
                dataset_name, 'global', **kwargs)


class EUniRepRegressionPredictor(BaseUniRepPredictor):
    """Evotuned UniRep + Ridge regression."""

    def __init__(self, dataset_name, rep_name='uniref100', **kwargs):
        super(EUniRepRegressionPredictor, self).__init__(
                dataset_name, rep_name, **kwargs)


class UniRepLLPredictor(BaseUniRepPredictor):
    """UniRep log likelihood."""

    def __init__(self, dataset_name, rep_name, reg_coef=1e-8, **kwargs):
        super(UniRepLLPredictor, self).__init__(
                dataset_name, rep_name, reg_coef=reg_coef, **kwargs)
        path_prefix = ''
        seqs_path = path_prefix + os.path.join('data', dataset_name, 'seqs.fasta')
        seqs = read_fasta(seqs_path)
        id2seq = pd.Series(index=np.arange(len(seqs)), data=seqs, name='seq')

        data_path = path_prefix + os.path.join('inference', dataset_name,
                'eUniRep', 'pll.csv')
        ll = pd.read_csv(data_path, index_col=0)
        ll['id'] = ll.index.to_series().apply(
                lambda x: int(x.replace('id_', '')))
        
        ll = ll.join(id2seq, on='id', how='left')

        self.seq2score_dict = dict(zip(ll.seq, ll.pll))

    def seq2score(self, seqs):
        scores = np.array([self.seq2score_dict.get(s, 0.0) for s in seqs])
        return scores

    def seq2feat(self, seqs):
        return self.seq2score(seqs)[:, None].reshape(len(seqs),-1)

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)

class GUniRepLLPredictor(UniRepLLPredictor):
    def __init__(self, dataset_name, **kwargs):
        super(GUniRepLLPredictor, self).__init__(dataset_name,
                'global', **kwargs)


class EUniRepLLPredictor(UniRepLLPredictor):
    def __init__(self, dataset_name, rep_name='uniref100', **kwargs):
        super(EUniRepLLPredictor, self).__init__(dataset_name,
                rep_name, **kwargs)
