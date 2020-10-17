#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: collect_stats.py
# --- Creation Date: 17-10-2020
# --- Last Modified: Sun 18 Oct 2020 00:46:46 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Collect evaluation results from all models.
"""

import argparse
import os
import pdb
import glob
import scipy

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

TPL_NAME = 'collected-tpl-mean.csv'
TPL_MEAN = 'avg_tpl.mean'
TPL_ACT = 'n_active_dims.mean'
SUPERVISED_ENTRIES = {
    # 'collected-tpl-mean.csv': ['avg_tpl.mean', 'n_active_dims.mean'],
    'collected-mig-mean.csv': 'discrete_mig.mean',
    'collected-dci-mean.csv': 'disentanglement.mean',
    'collected-factor_vae_metric-mean.csv': 'eval_accuracy.mean',
    'collected-beta_vae_sklearn-mean.csv': 'eval_accuracy.mean'
}
GOOD_THRESH = {
    'collected-mig-mean.csv': 0.10,
    'collected-dci-mean.csv': 0.25,
    'collected-factor_vae_metric-mean.csv': 0.65,
    'collected-beta_vae_sklearn-mean.csv': 0.75
}
BRIEF = {
    'collected-tpl-mean.csv': 'TPL',
    'collected-mig-mean.csv': 'MIG',
    'collected-dci-mean.csv': 'DCI',
    'collected-factor_vae_metric-mean.csv': 'FVM',
    'collected-beta_vae_sklearn-mean.csv': 'BVM'
}


def spearman_correl(a, b):
    correl_score, _ = scipy.stats.spearmanr(a, b)
    return correl_score


def lasso_correl(a, b):
    lasso = Lasso(max_iter=10000, normalize=True)
    lasso.fit(a[:, np.newaxis], b)
    # print('lasso.coef_:', lasso.coef_)
    return lasso.coef_[0]


CORREL_F = {'Spearman': spearman_correl, 'Lasso': lasso_correl}


def get_model_dirs_and_names(old_model_dirs):
    model_dirs = []
    model_names = []
    for name in old_model_dirs:
        if os.path.isdir(name):
            model_dirs.append(name)
            model_names.append(os.path.basename(name)[:-4])
    return model_dirs, model_names


def get_metric_file_names(model_dir):
    files = glob.glob(os.path.join(model_dir, '*'))
    metric_file_names = []
    for file in files:
        if os.path.basename(file) in SUPERVISED_ENTRIES:
            metric_file_names.append(os.path.basename(file))
    return metric_file_names


def get_correlation_results(tpl_file, metric_file, correl_fn):
    tpl_df = pd.read_csv(tpl_file)
    other_df = pd.read_csv(metric_file)
    tpl_array = tpl_df.loc[:, TPL_MEAN].values
    other_array = other_df.loc[:, SUPERVISED_ENTRIES[os.path.basename(
        metric_file)]].values

    # Calculate overall correlation scores.
    correl_score_overall = correl_fn(tpl_array, other_array)

    # Calculate per-act-dim correlation scores.
    tpl_act_dims_array = tpl_df.loc[:, TPL_ACT].values
    unique_dims = np.unique(tpl_act_dims_array)
    col_scores_for_act_dims = []
    for act_dim in unique_dims:
        tpl_dim_mask = tpl_act_dims_array == act_dim
        n_samples = tpl_dim_mask.astype(int).sum()
        tpl_i_array = np.extract(tpl_dim_mask, tpl_array)
        other_i_array = np.extract(tpl_dim_mask, other_array)
        # correl_score_i, _ = scipy.stats.spearmanr(tpl_i_array, other_i_array)
        correl_score_i = correl_fn(tpl_i_array, other_i_array)
        col_scores_for_act_dims.append([correl_score_i, n_samples])

    # Calculate correlation scores for rank < 20%.
    temp = other_array.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(other_array))  # rank entries by metric
    ranks_mask = ranks < (0.6 * len(other_array))
    tpl_rank_array = np.extract(ranks_mask, tpl_array)
    other_rank_array = np.extract(ranks_mask, other_array)
    correl_score_rank = correl_fn(tpl_rank_array, other_rank_array)
    return correl_score_overall, col_scores_for_act_dims, unique_dims, \
        correl_score_rank, other_array


def save_scores_for_act_dims(col_scores_for_act_dims, act_dims, model_dir,
                             metric, correlation_type):
    for i, act_dim in enumerate(act_dims):
        with open(
                os.path.join(
                    model_dir, correlation_type + '_' + metric + '_' +
                    str(act_dim) + '.txt'), 'w') as f:
            f.write('score={0:.4f}, n={1}'.format(
                col_scores_for_act_dims[i][0], col_scores_for_act_dims[i][1]))


def save_scores(results, index, columns, args, prefix='overall'):
    df = pd.DataFrame(results, index=index, columns=columns)
    df.to_csv(
        os.path.join(
            args.parent_parent_dir, prefix + '_' + args.correlation_type +
            '_' + BRIEF[TPL_NAME] + '_vs_others.csv'))


def get_tpl_all_scores(model_dirs):
    tpl_all = None
    for model_dir in model_dirs:
        tpl_file = os.path.join(model_dir, TPL_NAME)
        tpl_df = pd.read_csv(tpl_file)
        tpl_array = tpl_df.loc[:, TPL_MEAN].values
        if tpl_all is None:
            tpl_all = tpl_array
        else:
            tpl_all = np.concatenate((tpl_all, tpl_array), axis=0)
    return tpl_all


def get_all_scores(tpl_all_scores, metrics_scores, correl_fn, metric_file_names):
    # temp = tpl_all_scores.argsort()
    # ranks = np.empty_like(temp)
    # ranks[temp] = np.arange(len(tpl_all_scores))  # rank entries by metric
    # ranks_mask = ranks < (0.1 * len(tpl_all_scores))
    # tpl_rank_array = np.extract(ranks_mask, tpl_all_scores)
    scores_all = []
    scores_rank = []
    for i, metric_scores in enumerate(metrics_scores):
        # Calculate all scores.
        scores_all_i = correl_fn(tpl_all_scores, metric_scores)
        scores_all.append(scores_all_i)

        ranks_mask = GOOD_THRESH[metric_file_names[i]] < metric_scores
        tpl_rank_array = np.extract(ranks_mask, tpl_all_scores)
        # Calculate correlation scores for rank < 20%.
        other_rank_array = np.extract(ranks_mask, metric_scores)
        score_rank_i = correl_fn(tpl_rank_array, other_rank_array)
        scores_rank.append(score_rank_i)
    return scores_all, scores_rank


def main():
    parser = argparse.ArgumentParser(
        description='Collect statistics from TPL and '
        'other evaluation scores.')
    parser.add_argument('--parent_parent_dir',
                        help='Directory of parent dir of models to evaluate.',
                        type=str,
                        default='/mnt/hdd/repo_results/test')
    parser.add_argument('--correlation_type',
                        help='Correlation type.',
                        type=str,
                        default='Spearman',
                        choices=['Spearman', 'Lasso'])
    args = parser.parse_args()
    model_dirs = glob.glob(os.path.join(args.parent_parent_dir, '*'))
    model_dirs, model_names = get_model_dirs_and_names(model_dirs)
    metric_file_names = get_metric_file_names(model_dirs[0])
    results_overall_ls = []
    results_rank_ls = []
    metrics_scores = [None] * len(metric_file_names)
    correl_fn = CORREL_F[args.correlation_type]
    for model_dir in model_dirs:
        tpl_file = os.path.join(model_dir, TPL_NAME)
        results_overall_ls.append([])
        results_rank_ls.append([])
        for i, metric in enumerate(metric_file_names):
            metric_file = os.path.join(model_dir, metric)
            col_score, col_scores_for_act_dims, act_dims, correl_score_rank, metric_scores_i = \
                get_correlation_results(tpl_file, metric_file, correl_fn)
            if metrics_scores[i] is None:
                metrics_scores[i] = metric_scores_i
            else:
                metrics_scores[i] = np.concatenate(
                    (metrics_scores[i], metric_scores_i), axis=0)
            results_overall_ls[-1].append(col_score)
            results_rank_ls[-1].append(correl_score_rank)
            save_scores_for_act_dims(col_scores_for_act_dims, act_dims,
                                     model_dir, BRIEF[metric],
                                     args.correlation_type)
            print('metrics_scores[i].shape:', metrics_scores[i].shape)
    tpl_all_scores = get_tpl_all_scores(model_dirs)
    scores_all, scores_rank = get_all_scores(tpl_all_scores, metrics_scores,
                                             correl_fn, metric_file_names)

    save_scores(results_overall_ls,
                model_names, [BRIEF[name] for name in metric_file_names],
                args,
                prefix='overall')
    save_scores(results_rank_ls,
                model_names, [BRIEF[name] for name in metric_file_names],
                args,
                prefix='rank')
    save_scores(np.array(scores_all)[np.newaxis, ...],
                ['all_models'], [BRIEF[name] for name in metric_file_names],
                args,
                prefix='all')
    save_scores(np.array(scores_rank)[np.newaxis, ...],
                ['all_models'], [BRIEF[name] for name in metric_file_names],
                args,
                prefix='all_rank')


if __name__ == "__main__":
    main()
