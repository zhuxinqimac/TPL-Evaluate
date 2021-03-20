# TPL-Evaluate

This repository contains the code for evaluating the
Traversal Perceptual Length (TPL)
on the pretrained 1,800 checkpoints provided in
[disentanglemen_lib](https://github.com/google-research/disentanglement_lib)
on DSprites dataset.

## Requirements

* Python == 3.6.8
* Numpy == 1.17.2.
* TensorFlow == 1.14.0
* gin-config == 0.2.1
* matplotlib == 3.1.1
* Please follow [disentanglemen_lib](https://github.com/google-research/disentanglement_lib)
to setup the disentanglement_lib environment.

## Evaluation
Put the pretrained checkpoints on DSprites dataset of a model (e.g. BetaVAE)
into a directory (e.g. pretrained_models/beta_vae). There are multiple
checkpoints for a model with different hyper-parameters and random seeds.
Each checkpoint is identified by a number (from 0 to 1799).
There are 300 checkpoints for a model.

To evaluate the TPL for a model (taking beta_vae as an example),
use the following script:
```
CUDA_VISIBLE_DEVICES=0 \
    python sweep_evaluate.py \
    --parent_dir pretrained_models/beta_vae \
    --overwrite False
```
This compute the TPL scores for each checkpoint in the
pretrained_models/beta_vae folder, and put the results in the
pretrained_models/beta_vae/N/metrics/tpl directory.

Use the same method to compute TPL for other models like factor_vae, dip-i,
dip-ii, beta_tc_vae, and annealed_vae.

Use the following script to collect each metric results (e.g. TPL)
for each model (e.g. beta_vae):
```
python collect_results.py \
    --results_dir pretrained_models/beta_vae \
    --metric tpl
```

Use the following script to plot the correlation figures between the TPL and
other metrics.
```
python collect_stats.py --parent_parent_dir pretrained_models
```

## Citation
```
@inproceedings{Xinqi_cvpr21,
author={Xinqi Zhu and Chang Xu and Dacheng Tao},
title={Where and What? Examining Interpretable Disentangled Representations},
booktitle={CVPR},
year={2021}
}
```
