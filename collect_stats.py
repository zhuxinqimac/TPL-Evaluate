#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: collect_stats.py
# --- Creation Date: 17-10-2020
# --- Last Modified: Tue 27 Oct 2020 23:15:16 AEDT
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
import matplotlib.pyplot as plt

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

def read_tpl_array(tpl_file):
    tpl_df = pd.read_csv(tpl_file)
    tpl_array = tpl_df.loc[:, TPL_MEAN].values
    tpl_act_array = tpl_df.loc[:, TPL_ACT].values
    return tpl_array, tpl_act_array

def read_metric_array(metric_file):
    other_df = pd.read_csv(metric_file)
    other_array = other_df.loc[:, SUPERVISED_ENTRIES[os.path.basename(
        metric_file)]].values
    return other_array

def get_permodel_correlation_results(tpl_file, metric_file, correl_fn):
    tpl_array, _ = read_tpl_array(tpl_file)
    other_array = read_metric_array(metric_file)

    # Calculate overall correlation scores.
    correl_score_permodel = correl_fn(tpl_array, other_array)
    return correl_score_permodel

def get_perdim_correlation_results(tpl_file, metric_file, correl_fn):
    tpl_array, tpl_act_dims_array = read_tpl_array(tpl_file)
    other_array = read_metric_array(metric_file)

    # Calculate per-act-dim correlation scores.
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
    return col_scores_for_act_dims, unique_dims

def get_neartplthresh_results(tpl_file, metric_file, correl_fn):
    tpl_array, _ = read_tpl_array(tpl_file)
    other_array = read_metric_array(metric_file)

    # Calculate correlation scores around threshold of TPL: 1.2 - 1.6
    act_tpl = []
    act_metric= []
    for i, tpl_i in enumerate(tpl_array):
        if tpl_i > 1.2 and tpl_i < 1.6:
            act_tpl.append(tpl_i)
            act_metric.append(other_array[i])
    correl_score_act = correl_fn(np.array(act_tpl), np.array(act_metric))
    return correl_score_act

def get_otherranktop_results(tpl_file, metric_file, correl_fn):
    tpl_array, _ = read_tpl_array(tpl_file)
    other_array = read_metric_array(metric_file)

    # Calculate correlation scores for rank < 20%.
    temp = other_array.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(other_array))  # rank entries by metric
    ranks_mask = ranks < (0.6 * len(other_array))
    tpl_rank_array = np.extract(ranks_mask, tpl_array)
    other_rank_array = np.extract(ranks_mask, other_array)
    correl_score_rank = correl_fn(tpl_rank_array, other_rank_array)
    return correl_score_rank

def get_new_idx_per_model(model_idx_dict, idx_argsort):
    '''
    model_idx_dict: {'factovae': [300, ... 600], ...}
    '''
    n_total = len(idx_argsort)
    ranks = np.empty_like(idx_argsort)
    ranks[idx_argsort] = np.arange(len(idx_argsort))
    new_dict = {}
    for k, v in model_idx_dict.items():
        new_dict[k] = ranks[np.array(v)]
    return new_dict

def plot_file_tpl_v_metric(tpl_file, metric_file, save_dir, metric_name):
    tpl_array, tpl_act_array = read_tpl_array(tpl_file)
    other_array = read_metric_array(metric_file)
    plot_array_tpl_v_metric(tpl_array, other_array, save_dir, metric_name)

def extract_array_by_actdim(tpl_array, tpl_act_array, other_array):
    unique_dims = np.unique(tpl_act_array)
    col_scores_for_act_dims = []
    perdim_tpl_dict = {}
    perdim_other_dict = {}
    for act_dim in unique_dims:
        tpl_dim_mask = tpl_act_array == act_dim
        n_samples = tpl_dim_mask.astype(int).sum()
        tpl_i_array = np.extract(tpl_dim_mask, tpl_array)
        other_i_array = np.extract(tpl_dim_mask, other_array)
        perdim_tpl_dict[act_dim] = tpl_i_array
        perdim_other_dict[act_dim] = other_i_array
    return perdim_tpl_dict, perdim_other_dict

def greater_than_4(act_array):
    return act_array > 4

def greater_than_3(act_array):
    return act_array > 3

def extract_array_by_dimcond(tpl_array, tpl_act_array, other_array, dimcond=greater_than_4):
    tpl_dim_mask = dimcond(tpl_act_array)
    n_samples = tpl_dim_mask.astype(int).sum()
    tpl_cond_array = np.extract(tpl_dim_mask, tpl_array)
    other_cond_array = np.extract(tpl_dim_mask, other_array)
    return tpl_cond_array, other_cond_array

def plot_file_tpl_v_metric_perdim(tpl_file, metric_file, save_dir, metric_name):
    tpl_array, tpl_act_array = read_tpl_array(tpl_file)
    other_array = read_metric_array(metric_file)
    perdim_tpl_dict, perdim_other_dict = extract_array_by_actdim(tpl_array, tpl_act_array, other_array)

    for act_dim, tpl_i_array in perdim_tpl_dict.items():
        other_i_array = perdim_other_dict[act_dim]
        plot_array_tpl_v_metric(tpl_i_array, other_i_array, save_dir, metric_name, prefix='act'+str(act_dim))

def plot_array_tpl_v_metric(tpl_array, other_array, save_dir, metric_name, model_idx_dict=None, prefix=''):
    corr_score = spearman_correl(tpl_array, other_array)
    idx_argsort = tpl_array.argsort()
    tmp_arange = np.arange(len(idx_argsort))
    # sorted_tpl_array = tpl_array[idx_argsort]
    model_wise_prefix = ''
    if model_idx_dict is not None:
        new_model_idx_dict = get_new_idx_per_model(model_idx_dict, idx_argsort)
        for k, v in new_model_idx_dict.items():
            # plt.plot(tmp_arange[v], other_array[model_idx_dict[k]], 'ro')
            plt.bar(tmp_arange[v], other_array[model_idx_dict[k]], label=k)
        plt.legend()
        model_wise_prefix = 'colored'
    else:
        sorted_other_array_bytpl = other_array[idx_argsort]
        plt.bar(np.arange(len(sorted_other_array_bytpl)), sorted_other_array_bytpl)
        # plt.plot(np.arange(len(sorted_other_array_bytpl)), sorted_other_array_bytpl)
    plt.xlabel('TPL score rank')
    plt.ylabel(metric_name)
    ax = plt.gca()
    plt.grid(True)
    plt.text(0.2, 0.9, 'Spearman coef=%0.3f' % corr_score, transform = ax.transAxes)
    plt.savefig(os.path.join(save_dir, prefix+model_wise_prefix+'tpl_v_'+metric_name+'.pdf'))
    plt.clf()

def save_scores_for_act_dims(col_scores_for_act_dims, act_dims, model_dir,
                             metric, correlation_type):
    for i, act_dim in enumerate(act_dims):
        with open(
                os.path.join(
                    model_dir, correlation_type + '_' + metric + '_' +
                    str(act_dim) + '.txt'), 'w') as f:
            print('saving in: ', model_dir, '; for dim: ', act_dim)
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
    tpl_act_all = None
    for model_dir in model_dirs:
        tpl_file = os.path.join(model_dir, TPL_NAME)
        # tpl_df = pd.read_csv(tpl_file)
        # tpl_array = tpl_df.loc[:, TPL_MEAN].values
        tpl_array, tpl_act_array = read_tpl_array(tpl_file)
        if tpl_all is None:
            tpl_all = tpl_array
            tpl_act_all = tpl_act_array
        else:
            tpl_all = np.concatenate((tpl_all, tpl_array), axis=0)
            tpl_act_all = np.concatenate((tpl_act_all, tpl_act_array), axis=0)
    return tpl_all, tpl_act_all


def get_all_scores(tpl_all_scores, metrics_scores, correl_fn, metric_file_names):
    scores_all = []
    for i, metric_scores in enumerate(metrics_scores):
        scores_all_i = correl_fn(tpl_all_scores, metric_scores)
        scores_all.append(scores_all_i)

    return scores_all

def get_all_neartplthresh_scores(tpl_all_scores, metrics_scores, correl_fn, metric_file_names):
    scores_all_act = []
    for i, metric_scores in enumerate(metrics_scores):
        act_tpl = []
        act_metric = []
        for j, tpl_j in enumerate(tpl_all_scores):
            # Calculate correlation scores around threshold of TPL: 1.2 - 1.6
            if tpl_j > 1.2 and tpl_j < 1.6:
                act_tpl.append(tpl_j)
                act_metric.append(metric_scores[j])
        scores_all_act_i = correl_fn(act_tpl, act_metric)
        scores_all_act.append(scores_all_act_i)
    return scores_all_act

def get_all_otherranktop_scores(tpl_all_scores, metrics_scores, correl_fn, metric_file_names):
    scores_rank = []
    for i, metric_scores in enumerate(metrics_scores):
        ranks_mask = GOOD_THRESH[metric_file_names[i]] < metric_scores
        tpl_rank_array = np.extract(ranks_mask, tpl_all_scores)
        other_rank_array = np.extract(ranks_mask, metric_scores)
        score_rank_i = correl_fn(tpl_rank_array, other_rank_array)
        scores_rank.append(score_rank_i)
    return scores_rank


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
    results_near_tpl_thresh_ls = []
    metrics_scores = [None] * len(metric_file_names)
    model_idx_for_metric = [{}] * len(metric_file_names)
    correl_fn = CORREL_F[args.correlation_type]
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        tpl_file = os.path.join(model_dir, TPL_NAME)
        results_overall_ls.append([])
        results_near_tpl_thresh_ls.append([])
        tpl_array, tpl_act_array = read_tpl_array(tpl_file)
        for i, metric in enumerate(metric_file_names):
            metric_file = os.path.join(model_dir, metric)
            other_array = read_metric_array(metric_file)
            metric_name_i = BRIEF[metric]

            # All plot
            plot_array_tpl_v_metric(tpl_array, other_array, model_dir, metric_name_i)
            # Per-dim plot
            perdim_tpl_dict, perdim_other_dict = extract_array_by_actdim(tpl_array, tpl_act_array, other_array)
            for act_dim, tpl_i_array in perdim_tpl_dict.items():
                other_i_array = perdim_other_dict[act_dim]
                plot_array_tpl_v_metric(tpl_i_array, other_i_array, model_dir, metric_name_i, prefix='act'+str(act_dim))
            # Dim-conditioned plot
            tpl_cond_array, other_cond_array = extract_array_by_dimcond(tpl_array, tpl_act_array, other_array, dimcond=greater_than_4)
            plot_array_tpl_v_metric(tpl_cond_array, other_cond_array, model_dir, metric_name_i, prefix='act>4')
            tpl_cond_array, other_cond_array = extract_array_by_dimcond(tpl_array, tpl_act_array, other_array, dimcond=greater_than_3)
            plot_array_tpl_v_metric(tpl_cond_array, other_cond_array, model_dir, metric_name_i, prefix='act>3')

            col_score = get_permodel_correlation_results(tpl_file, metric_file, correl_fn)
            near_tpl_thresh_score = get_neartplthresh_results(tpl_file, metric_file, correl_fn)

            results_overall_ls[-1].append(col_score)
            results_near_tpl_thresh_ls[-1].append(near_tpl_thresh_score)

            # Collect overall array for each metric
            if metrics_scores[i] is None:
                model_idx_for_metric[i][model_name] = np.arange(len(other_array))
                metrics_scores[i] = other_array
            else:
                tmp_len = len(metrics_scores[i])
                model_idx_for_metric[i][model_name] = np.arange(tmp_len, tmp_len + len(other_array))
                metrics_scores[i] = np.concatenate(
                    (metrics_scores[i], other_array), axis=0)

            # # Collect perdim scores
            # col_scores_for_act_dims, act_dims = get_perdim_correlation_results(tpl_file, metric_file, correl_fn)
            # save_scores_for_act_dims(col_scores_for_act_dims, act_dims,
                                     # model_dir, metric_name_i,
                                     # args.correlation_type)
            # print('metrics_scores[i].shape:', metrics_scores[i].shape)
    tpl_all_scores, tpl_act_all = get_tpl_all_scores(model_dirs)
    scores_all = get_all_scores(tpl_all_scores, metrics_scores, correl_fn, metric_file_names)
    scores_all_rank = get_all_otherranktop_scores(tpl_all_scores, metrics_scores, correl_fn, metric_file_names)
    scores_all_neartplthresh = get_all_neartplthresh_scores(tpl_all_scores, metrics_scores, correl_fn, metric_file_names)
    for i, metric_scores in enumerate(metrics_scores):
        metric_name_i = BRIEF[metric_file_names[i]]
        # All plot
        plot_array_tpl_v_metric(tpl_all_scores, metric_scores, args.parent_parent_dir, metric_name_i)
        plot_array_tpl_v_metric(tpl_all_scores, metric_scores, args.parent_parent_dir, metric_name_i, model_idx_dict=model_idx_for_metric[i])
        # Per-dim plot
        perdim_tpl_dict, perdim_other_dict = extract_array_by_actdim(tpl_all_scores, tpl_act_all, metric_scores)
        for act_dim, tpl_i_array in perdim_tpl_dict.items():
            other_i_array = perdim_other_dict[act_dim]
            plot_array_tpl_v_metric(tpl_i_array, other_i_array, args.parent_parent_dir, metric_name_i, prefix='act'+str(act_dim))
        # Dim-conditioned plot
        tpl_cond_array, other_cond_array = extract_array_by_dimcond(tpl_all_scores, tpl_act_all, metric_scores, dimcond=greater_than_4)
        # model_idx_dict_cond = extract_model_idx_dict_by_dimcond(model_idx_dic, tpl_act_all, dimcond=greater_than_4)
        plot_array_tpl_v_metric(tpl_cond_array, other_cond_array, args.parent_parent_dir, metric_name_i, prefix='act>4')
        tpl_cond_array, other_cond_array = extract_array_by_dimcond(tpl_all_scores, tpl_act_all, metric_scores, dimcond=greater_than_3)
        # model_idx_dict_cond = extract_model_idx_dict_by_dimcond(model_idx_dic, tpl_act_all, dimcond=greater_than_3)
        plot_array_tpl_v_metric(tpl_cond_array, other_cond_array, args.parent_parent_dir, metric_name_i, prefix='act>3')

    save_scores(results_overall_ls,
                model_names, [BRIEF[name] for name in metric_file_names],
                args,
                prefix='overall')
    save_scores(results_near_tpl_thresh_ls,
                model_names, [BRIEF[name] for name in metric_file_names],
                args,
                prefix='neartplthresh')
    save_scores(np.array(scores_all)[np.newaxis, ...],
                ['all_models'], [BRIEF[name] for name in metric_file_names],
                args,
                prefix='all')
    save_scores(np.array(scores_all_rank)[np.newaxis, ...],
                ['all_models'], [BRIEF[name] for name in metric_file_names],
                args,
                prefix='all_rank')
    save_scores(np.array(scores_all_neartplthresh)[np.newaxis, ...],
                ['all_models'], [BRIEF[name] for name in metric_file_names],
                args,
                prefix='all_neartplthresh')


if __name__ == "__main__":
    main()
