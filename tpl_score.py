#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: tpl_score.py
# --- Creation Date: 13-10-2020
# --- Last Modified: Tue 13 Oct 2020 22:56:50 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Implementation of the TPL score.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import pickle
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'disentanglement_lib'))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'stylegan2'))
import dnnlib.tflib
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
from six.moves import range
from scipy import stats
import gin.tf


@gin.configurable("tpl_score",
                  blacklist=[
                      "ground_truth_data", "generator_function",
                      "random_state", "activation_str", "latent_size",
                      "artifact_dir"
                  ])
def compute_tpl_score(ground_truth_data,
                      generator_function,
                      random_state,
                      activation_str,
                      latent_size,
                      artifact_dir=None,
                      batch_size=gin.REQUIRED,
                      num_traversals=gin.REQUIRED,
                      num_samples_per_dim=gin.REQUIRED,
                      active_thresh=gin.REQUIRED):
    """Computes the FactorVAE disentanglement metric.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    generator_function: Function that takes latent code as input and
      outputs an image.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    latent_size: Latent code size.
    batch_size: Number of points to be used to in a GPU.
    num_traversals: Number of traversals to sample.
    num_samples_per_dim: Number of samples per dim.
    active_thresh: Threshold for determining active dims.

  Returns:
    Dictionary with scores:
      avg_tpl_dim: Mean TPL score for each dim.
      avg_tpl: Overall mean TPL score.
      active_mask: Mask of which dims are active.
      active_distances: Latent distances of active dims.
      active_stds: Latent stds of active dims.
      n_active_dims: Number of active dims.
  """
    del ground_truth_data
    del artifact_dir
    dnnlib.tflib.init_tf()
    distance_measure = load_pkl(
        '.stylegan2-cache/vgg16_zhang_perceptual.pkl'
        # 'http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl'
    )
    if activation_str == "'logits'":
        activation = sigmoid
    elif activation_str == "'tanh'":
        activation = tanh
    else:
        raise ValueError(
            "Activation function  could not be infered from gin config.")
    tpl_dim_ls = []
    for i in range(num_traversals):
        sample = random_state.normal(size=(1, latent_size))
        # factor_index = random_state.randint(ground_truth_data.num_factors)
        tpl_dim = compute_tpl_for(
            sample, generator_function, batch_size, num_samples_per_dim,
            latent_size, activation,
            distance_measure)  # np.array of [latent_size]
        tpl_dim_ls.append(tpl_dim)
    tpl_dim_np = np.array(tpl_dim_ls)  # np.array of [n_trav, latent_size]
    avg_tpl_dim = np.mean(tpl_dim_np, axis=0)  # np.array of [latent_size]
    std_tpl_dim = np.std(tpl_dim_np, axis=0)  # np.array of [latent_size]

    active_mask = avg_tpl_dim > active_thresh
    active_distances = np.extract(active_mask, avg_tpl_dim)
    active_stds = np.extract(active_mask, std_tpl_dim)
    avg_tpl = np.sum(active_distances)
    n_active_dims = active_mask.astype(int).sum()

    scores_dict = {}
    scores_dict['avg_tpl_dim'] = avg_tpl_dim.tolist()
    scores_dict['avg_tpl'] = avg_tpl
    scores_dict['active_mask'] = active_mask.tolist()
    scores_dict['active_distances'] = active_distances.tolist()
    scores_dict['active_stds'] = active_stds.tolist()
    scores_dict['n_active_dims'] = n_active_dims
    print('scores_dict:', scores_dict)
    return scores_dict


def compute_tpl_for(sample, generator_function, batch_size,
                    num_samples_per_dim, latent_size, activation,
                    distance_measure):
    '''
    Return: np.array of [latent_size]
    '''
    tpl_sample_dim_ls = []
    for i in range(latent_size):
        samples_traversal = np.tile(sample, [num_samples_per_dim, 1])
        samples_traversal[:, i] = np.linspace(-2., 2., num=num_samples_per_dim)
        raw_imgs_traversal = generator_function(
            samples_traversal)  # size: [b, h, w, c]
        imgs_traversal = activation(raw_imgs_traversal)
        j = 0
        traversal_score = 0
        while j < num_samples_per_dim:
            b_j = min(batch_size, num_samples_per_dim)
            cur_imgs_traversal = imgs_traversal[j:j + b_j, ...]
            j += b_j
            traversal_score += measure_distance(cur_imgs_traversal,
                                                distance_measure)
        tpl_sample_dim_ls.append(traversal_score)
    return np.array(tpl_sample_dim_ls)


def measure_distance(imgs_traversal, distance_measure):
    images = np.transpose(imgs_traversal, [0, 3, 1, 2])  # bhwc -> bchw
    images = images * 255  # [0, 1] -> [0, 255]
    v = get_return_v(distance_measure.run(images[:-1, ...], images[1:, ...]),
                     1)
    return v.sum()


def sigmoid(x):
    return stats.logistic.cdf(x)


def tanh(x):
    return np.tanh(x) / 2. + .5


def get_return_v(x, topk=1):
    if (not isinstance(x, tuple)) and (not isinstance(x, list)):
        return x if topk == 1 else tuple([x] + [None] * (topk - 1))
    if topk > len(x):
        return tuple(list(x) + [None] * (topk - len(x)))
    else:
        if topk == 1:
            return x[0]
        else:
            return tuple(x[:topk])


def load_pkl(file_or_url):
    with open(file_or_url, 'rb') as file:
        return pickle.load(file, encoding='latin1')
