import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from utils import seqs_to_onehot, get_wt_seq, read_fasta, seq2effect, mutant2seq
from predictors.base_predictors import BaseRegressionPredictor

# EVMutation imports
from couplings_model import CouplingsModel


class EVPredictor(BaseRegressionPredictor):
    """plmc mutation effect prediction."""

    def __init__(self, dataset_name, model_name='uniref100',
        reg_coef=1e-8, ignore_gaps=False, **kwargs):
        super(EVPredictor, self).__init__(dataset_name, reg_coef=reg_coef,
                **kwargs)

        path_prefix = ''
        seqs_path = path_prefix + os.path.join('data', dataset_name, 'seqs.fasta')
        seqs = read_fasta(seqs_path)
        id2seq = pd.Series(index=np.arange(len(seqs)), data=seqs, name='seq')

        data_path = path_prefix + os.path.join('inference', dataset_name,
                'ev', 'pll.csv')
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
