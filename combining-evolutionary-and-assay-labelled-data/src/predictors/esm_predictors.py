import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from utils import seqs_to_onehot, read_fasta, load_rows_by_numbers
from predictors.base_predictors import BaseRegressionPredictor
import pickle
import glob



class DDGPredictor(BaseRegressionPredictor):
    def __init__(self, dataset_name, reg_coef=1e-8, path_prefix='',
            **kwargs):
        super(DDGPredictor, self).__init__(dataset_name, reg_coef, Ridge)
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
class ESMPredictor(BaseRegressionPredictor):
    """ESM likelihood as features in regression."""

    def __init__(self, dataset_name, rep_name, reg_coef=1e-8, path_prefix='',
            **kwargs):
        super(ESMPredictor, self).__init__(dataset_name, reg_coef, Ridge)

        seqs_path = path_prefix + os.path.join('data', dataset_name, 'seqs.fasta')
        seqs = read_fasta(seqs_path)
        id2seq = pd.Series(index=np.arange(len(seqs)), data=seqs, name='seq')

        esm_data_path = path_prefix + os.path.join('inference', dataset_name,
                'esm', 'pll.csv')
        ll = pd.read_csv(esm_data_path, index_col=0)
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


class GlobalESMPredictor(ESMPredictor):
    def __init__(self, dataset_name, **kwargs):
        super(GlobalESMPredictor, self).__init__(
            dataset_name, 'global', **kwargs)


class EvotunedESMPredictor(ESMPredictor):
    def __init__(self, dataset_name, rep_name='uniref100', **kwargs):
        super(EvotunedESMPredictor, self).__init__(
            dataset_name, rep_name, **kwargs)


class ESMRegressionPredictor(BaseRegressionPredictor):
    """Regression on ESM representation."""

    def __init__(self, dataset_name, rep_name, reg_coef=1.0, **kwargs):
        super(ESMRegressionPredictor, self).__init__(dataset_name, reg_coef,
                Ridge, **kwargs)
        self.load_rep(dataset_name, rep_name)

    def load_rep(self, dataset_name, rep_name):
        self.rep_path = os.path.join('inference', dataset_name, 'esm', 
                rep_name, 'rep.npy*')
        self.seq_path = os.path.join('data', dataset_name, 'seqs.fasta')
        self.seqs = read_fasta(self.seq_path)
        self.seq2id = dict(zip(self.seqs, range(len(self.seqs))))

    def seq2feat(self, seqs):
        """Look up representation by sequences."""
        ids = [self.seq2id[s] for s in seqs]
        return load_rows_by_numbers(self.rep_path, ids)


class EvotunedESMRegressionPredictor(ESMRegressionPredictor):
    def __init__(self, dataset_name, **kwargs):
        super(EvotunedESMRegressionPredictor, self).__init__(dataset_name,
                'uniref100', **kwargs)


class GlobalESMRegressionPredictor(ESMRegressionPredictor):
    def __init__(self, dataset_name, **kwargs):
        super(GlobalESMRegressionPredictor, self).__init__(dataset_name,
                'global', **kwargs)


