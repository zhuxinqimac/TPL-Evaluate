#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: sweep_evaluate.py
# --- Creation Date: 13-10-2020
# --- Last Modified: Tue 20 Oct 2020 01:47:05 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Evaluate a directory of models by our proposed TPL score.
"""

import argparse
import os
import pdb

import evaluate_with_generator as evaluate
import gin.tf
import glob


def main():
    parser = argparse.ArgumentParser(description='TPL score evaluation.')
    parser.add_argument('--parent_dir',
                        help='Directory of models to evaluate.',
                        type=str,
                        default='/mnt/hdd/repo_results/test')
    parser.add_argument('--eval_gin',
                        help='Gin for TPL score.',
                        type=str,
                        default='tpl_config.gin')
    parser.add_argument('--overwrite',
                        help='Whether to overwrite output directory.',
                        type=_str_to_bool,
                        default=False)
    args = parser.parse_args()
    model_dirs = glob.glob(os.path.join(args.parent_dir, '*'))
    for path in model_dirs:
        if not os.path.isdir(path):
            continue
        model_path = os.path.join(path, "model")
        result_path = os.path.join(path, "metrics", "tpl")
        if not args.overwrite and os.path.isdir(os.path.join(result_path, 'results')):
            continue
        else:
            evaluate.evaluate_with_gin_with_generator(model_path, result_path,
                                                      args.overwrite, [args.eval_gin])


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    main()
