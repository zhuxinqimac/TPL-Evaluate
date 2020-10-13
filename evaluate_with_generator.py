#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: evaluate_with_generator.py
# --- Creation Date: 12-10-2020
# --- Last Modified: Tue 13 Oct 2020 19:02:50 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Evaluation protocol (generator version) to compute metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                 'disentanglement_lib'))
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.utils import results
import tpl_score

import inspect
import os
import time
import warnings

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

import gin.tf


def evaluate_with_gin_with_generator(model_dir,
                                     output_dir,
                                     overwrite=False,
                                     gin_config_files=None,
                                     gin_bindings=None):
    """Evaluate a generator based on the provided gin configuration.

  This function will set the provided gin bindings, call the evaluate()
  function and clear the gin config. Please see the evaluate() for required
  gin bindings.

  Args:
    model_dir: String with path to directory where the model is saved.
    output_dir: String with the path where the evaluation should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    gin_config_files: List of gin config files to load.
    gin_bindings: List of gin bindings to use.
  """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    evaluate_with_generator(model_dir, output_dir, overwrite)
    gin.clear_config()


@gin.configurable("evaluation_with_generator",
                  blacklist=["model_dir", "output_dir", "overwrite"])
def evaluate_with_generator(model_dir,
                            output_dir,
                            overwrite=False,
                            evaluation_fn=gin.REQUIRED,
                            random_seed=gin.REQUIRED,
                            name=""):
    """Loads a generator TFHub module and computes disentanglement metrics.

  Args:
    model_dir: String with path to directory where the representation function
      is saved.
    output_dir: String with the path where the results should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    evaluation_fn: Function used to evaluate the representation (see metrics/
      for examples).
    random_seed: Integer with random seed used for training.
    name: Optional string with name of the metric (can be used to name metrics).
  """
    # Delete the output directory if it already exists.
    if tf.gfile.IsDirectory(output_dir):
        if overwrite:
            tf.gfile.DeleteRecursively(output_dir)
        else:
            raise ValueError(
                "Directory already exists and overwrite is False.")

    # Set up time to keep track of elapsed time in results.
    experiment_timer = time.time()

    # Automatically set the proper data set if necessary. We replace the active
    # gin config as this will lead to a valid gin config file where the data set
    # is present.
    gin_config_file = os.path.join(model_dir, "results", "gin", "train.gin")
    gin_dict = results.gin_dict(gin_config_file)
    if gin.query_parameter("dataset.name") == "auto":
        # Obtain the dataset name from the gin config of the previous step.
        with gin.unlock_config():
            gin.bind_parameter("dataset.name",
                               gin_dict["dataset.name"].replace("'", ""))
    dataset = named_data.get_named_ground_truth_data()

    activation_str = gin_dict["reconstruction_loss.activation"]
    latent_size = gin_dict["encoder.num_latent"]

    module_path = os.path.join(model_dir, "tfhub")
    with hub.eval_function_for_module(module_path) as f:

        def _generator_function(x):
            """Computes representation vector for input images."""
            output = f(dict(latent_vectors=x),
                       signature="decoder",
                       as_dict=True)
            return np.array(output["images"])

        # Computes scores of the representation based on the evaluation_fn.
        if _has_kwarg_or_kwargs(evaluation_fn, "artifact_dir"):
            artifact_dir = os.path.join(model_dir, "artifacts", name)
            results_dict = evaluation_fn(
                dataset,
                _generator_function,
                random_state=np.random.RandomState(random_seed),
                activation_str=activation_str,
                artifact_dir=artifact_dir,
                latent_size=latent_size)
        else:
            # Legacy code path to allow for old evaluation metrics.
            warnings.warn(
                "Evaluation function does not appear to accept an"
                " `artifact_dir` argument. This may not be compatible with "
                "future versions.", DeprecationWarning)
            results_dict = evaluation_fn(
                dataset,
                _generator_function,
                random_state=np.random.RandomState(random_seed),
                activation_str=activation_str,
                latent_size=latent_size)

    # Save the results (and all previous results in the pipeline) on disk.
    original_results_dir = os.path.join(model_dir, "results")
    results_dir = os.path.join(output_dir, "results")
    results_dict["elapsed_time"] = time.time() - experiment_timer
    results.update_result_directory(results_dir, "evaluation", results_dict,
                                    original_results_dir)


def _has_kwarg_or_kwargs(f, kwarg):
    """Checks if the function has the provided kwarg or **kwargs."""
    # For gin wrapped functions, we need to consider the wrapped function.
    if hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    (args, _, kwargs, _, _, _, _) = inspect.getfullargspec(f)
    if kwarg in args or kwargs is not None:
        return True
    return False
