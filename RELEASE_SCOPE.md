# Release scope

This file defines the boundary of the current S3AD open-source release candidate.

## Included

- S3AD and its evaluation metrics;
- S3AD, CS-LSTMs, KAN-AD, and DLinear adapters;
- model implementations required by the clean benchmark;
- data loading, splitting, normalization, and evaluation utilities;
- benchmark datasets used by the clean experiments;
- the clean benchmark entry point `experiments/exp_clean.py`;
- installation and dataset-placement documentation.

## Deliberately withheld

- spike, trend, variance, and frequency perturbation scripts;
- point-missing and block-missing scripts;
- robustness experiment runners and robustness result files;
- internal result-processing scripts;
- paper drafts, review materials, and local datasets.

## Current status

The repository is intended to publish the core code and benchmark datasets. After paper acceptance, the authors will add a license, verify dataset redistribution terms, and decide whether the robustness code should also be released.
