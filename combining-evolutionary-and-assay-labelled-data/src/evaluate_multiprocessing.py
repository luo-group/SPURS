'''
Inner loop in evaluating predictors with multiple processors.
See also evaluate.py.
'''
import functools
import logging
import os

import numpy as np
import pandas as pd

from predictors import get_predictor_cls, BoostingPredictor, JointPredictor
from utils.metric_utils import spearman, topk_mean, r2, hit_rate, aucroc, ndcg
from utils.io_utils import load_data_split, get_wt_log_fitness, get_log_fitness_cutoff
from utils.data_utils import dict2str


MAX_N_TEST=10000


def evaluate_predictor(dataset_name, predictor_name, joint_training,
        n_train, metric_topk, max_n_mut, train_on_single, ignore_gaps,
        seed, predictor_params, outpath):
    print(f'----- predictor {predictor_name}, seed {seed} -----')
    print(dataset_name)
    outpath = f'{outpath}-{os.getpid()}'  # each process writes to its own file
    data = load_data_split(dataset_name, split_id=-1,
            ignore_gaps=ignore_gaps)

    predictor_cls = get_predictor_cls(predictor_name)
    if len(predictor_cls) == 1:
        predictor = predictor_cls[0](dataset_name, **predictor_params)
    elif joint_training:
        predictor = JointPredictor(dataset_name, predictor_cls,
            predictor_name, **predictor_params)
    else:
        predictor = BoostingPredictor(dataset_name, predictor_cls,
            **predictor_params)

    test = data.sample(frac=0.2, random_state=seed)
    if len(test) > MAX_N_TEST:
        test = test.sample(n=MAX_N_TEST, random_state=seed)
    test = test.copy()
    train = data.drop(test.index)
    if train_on_single and 'n_mut' in data.columns:
        train = train[train.n_mut <= 1]
    assert len(train) >= n_train, 'not enough training data'

    if n_train == 0:
        test['pred'] = predictor.predict_unsupervised(test.seq.values)
    elif n_train == -1:   # -1 indicates 80/20 split
        n_train = len(train)
        predictor.train(train.seq.values, train.log_fitness.values)
        test['pred'] = predictor.predict(test.seq.values)
    else:

        train = predictor.select_training_data(train, n_train)
        assert len(train) == n_train, (
            f'expected {n_train} train examples, received {len(train)}')
        predictor.train(train.seq.values, train.log_fitness.values)
        test['pred'],vae_score,ddg_score = predictor.predict(test.seq.values,ret_values=True)

    metric_fns = {
        'spearman': spearman,
        'ndcg': ndcg,
        #'r2': r2,
        'topk_mean': functools.partial(
            topk_mean, topk=metric_topk),
        #'hit_rate_wt': functools.partial(
        #    hit_rate, y_ref=get_wt_log_fitness(dataset_name),
        #    topk=metric_topk),
        #'hit_rate_bt': functools.partial(
        #    hit_rate, y_ref=train.log_fitness.max(), topk=metric_topk),
        #'aucroc': functools.partial(
        #    aucroc, y_cutoff=get_log_fitness_cutoff(dataset_name)),
    }
    pred_test_path = outpath[:-9]+'result'+str(os.getpid())+predictor_name+'.npy'

    results_dict = {k: mf(test.pred.values, test.log_fitness.values)
            for k, mf in metric_fns.items()}
    if 'n_mut' in data.columns:
        max_n_mut = min(data.n_mut.max(), max_n_mut)
        for j in range(1, max_n_mut+1):
            y_pred = test[test.n_mut == j].pred.values
            y_true = test[test.n_mut == j].log_fitness.values
            results_dict.update({
                    f'{k}_{j}mut': mf(y_pred, y_true)
                    for k, mf in metric_fns.items()})
    results_dict.update({
        'dataset': dataset_name,
        'predictor': predictor_name,
        'n_train': n_train,
        'topk': metric_topk,
        'seed': seed,
        'predictor_params': dict2str(predictor_params),
    })
    results = pd.DataFrame(columns=sorted(results_dict.keys()))
    results = results.append(results_dict, ignore_index=True)

    if os.path.exists(outpath):
        results.to_csv(outpath, mode='a', header=False, index=False,
                columns=sorted(results.columns.values))
    else:
        results.to_csv(outpath, mode='w', index=False,
                columns=sorted(results.columns.values))

    return results


def run_from_queue(worker_id, queue):
    while True:
        args = queue.get()
        try:
            evaluate_predictor(*args)
        except Exception as e:
            logging.error("ERROR: %s", str(e))
            logging.exception(e)
            queue.task_done()
        queue.task_done()

