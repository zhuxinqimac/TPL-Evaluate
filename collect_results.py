#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: collect_results.py
# --- Creation Date: 08-09-2020
# --- Last Modified: Wed 07 Oct 2020 15:37:38 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Collect results.
"""

import os
import json
import numpy as np
import argparse
import pandas as pd
from collections import OrderedDict

METRICS_TEMPLATE = {
    'beta_vae_sklearn': {
        "train_accuracy": None,
        "eval_accuracy": None
    },
    'dci': {
        "informativeness_train": None,
        "informativeness_test": None,
        "disentanglement": None,
        "completeness": None
    },
    'downstream_task_boosted_trees': {},
    'factor_vae_metric': {
        "train_accuracy": None,
        "eval_accuracy": None,
        # "num_active_dims": None # disentanglement_lib wrong implementation.
    },
    'mig': {
        "discrete_mig": None
    },
    'modularity_explicitness': {
        "modularity_score": None,
        "explicitness_score_train": None,
        "explicitness_score_test": None
    },
    'sap_score': {
        "SAP_score": None
    },
    'unsupervised': {
        "gaussian_total_correlation": None,
        "gaussian_wasserstein_correlation": None,
        "gaussian_wasserstein_correlation_norm": None,
        "mutual_info_score": None
    }
}


def get_mean_std_for_config(v_ls, target):
    '''
    v_ls: [{'eval':0.8, ..}, {'eval': 0.7, ...}, ...]
    target: 'eval'
    '''
    pure_ls = []
    for item in v_ls:
        if item is not None:
            pure_ls.append(item[target])
    return (None, None) if len(pure_ls) == 0 else (np.mean(pure_ls),
                                                   np.std(pure_ls))


def count_samples(x):
    x = list(filter(None, x))
    return len(x)


def get_moments(res_dict, template):
    '''
    Args: result dict for each config and seed:
        {'0_0_0_0': [{'eval':0.8}, {'eval': 0.7}, ...]}
        template of collected results:
        {'eval': None, ...}
    Return: mean and std of each config:
        {'0_0_0_0': {'eval.mean': 0.75, 'eval.std': 0.05}, ...}
    '''
    res_dict_moments = {}
    for k, v in res_dict.items():
        res_dict_moments[k] = {}
        for res_k in template.keys():
            res_dict_moments[k][res_k+'.mean'], \
                res_dict_moments[k][res_k+'.std'] \
                = get_mean_std_for_config(v, res_k)
        res_dict_moments[k]['n_samples'] = count_samples(v)
    return res_dict_moments


def get_metric_result(subdir, metric, representation):
    result_json = os.path.join(subdir, 'metrics', representation, metric,
                               'results/json/evaluation_results.json')
    if os.path.exists(result_json):
        with open(result_json, 'r') as f:
            data = json.load(f)
        return data
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description='Project description.')
    parser.add_argument('--results_dir',
                        help='Results directory.',
                        type=str,
                        default='/mnt/hdd/repo_results/Ramiel/sweep')
    parser.add_argument('--metric',
                        help='Name of the collect metric.',
                        type=str,
                        default='factor_vae_metric',
                        choices=[
                            'beta_vae_sklearn', 'dci',
                            'downstream_task_boosted_trees',
                            'factor_vae_metric', 'mig',
                            'modularity_explicitness', 'sap_score',
                            'unsupervised'
                        ])
    parser.add_argument('--representation',
                        help='Representation used.',
                        type=str,
                        default='mean',
                        choices=['mean', 'sampled'])
    # parser.add_argument('--overwrite',
    # help='Whether to overwrite output directory.',
    # type=_str_to_bool,
    # default=False)
    args = parser.parse_args()
    subdirs = os.listdir(args.results_dir)
    res_dict = {}
    key_template = METRICS_TEMPLATE[args.metric]
    for subdir in subdirs:
        sub_path = os.path.join(args.results_dir, subdir)
        if not os.path.isdir(sub_path):
            continue
        parse_subdir = subdir.split('-')
        if len(parse_subdir) >= 7:
            hyps = '-'.join(parse_subdir[1:5]+parse_subdir[6:])
            seed = parse_subdir[5]
        else:
            hyps = '-'.join(parse_subdir[1:-1])
            seed = parse_subdir[-1]
        if hyps not in res_dict:
            res_dict[hyps] = [None] * 10
        # get result for this seed, a dictionary.
        res_dict[hyps][int(seed)] = get_metric_result(sub_path, args.metric,
                                                      args.representation)
    # {'0_0_0_0': {'eval.mean': 0.75, 'eval.std': 0.05, 'n_samples': 2}, ...}
    res_dict = get_moments(res_dict, key_template)
    col_heads = ['_config'] + list(res_dict[list(res_dict.keys())[0]].keys())
    col_dicts = {k: [] for k in col_heads}
    for k, v in res_dict.items():
        col_dicts['_config'].append(k)
        for k in col_dicts.keys():
            if k != '_config':
                col_dicts[k].append(v[k])
    new_results = OrderedDict(sorted(col_dicts.items()))
    results_df = pd.DataFrame(new_results)
    print('results_df:', results_df)
    results_df.to_csv(os.path.join(
        args.results_dir,
        'collected-' + args.metric + '-' + args.representation + '.csv'),
                      na_rep='-',
                      index=False,
                      float_format='%.3f')


if __name__ == "__main__":
    main()
