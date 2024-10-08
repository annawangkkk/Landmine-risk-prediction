# Landmine-risk-prediction

Repo of [RELand: Risk Estimation of Landmines via Interpretable Invariant Risk Minimization](https://dl.acm.org/doi/full/10.1145/3648437).

# Requirements

Required packages:
```
pandas
numpy
sklearn
pytorch
lightgbm
pytorch_tabnet
```

# Dataset

The dataset used for experiments in paper is in `/processed_dataset` folder.

# Validation

We provide three validation methods in our paper in `/train_val_stream` folder. 
- `blockCV`: blockCV method
- `bolivar`: blockV method
- `transfer`: transferCV method

# Running Experiments
We provide an example in `bash run_reland.sh`. Some configurations include:
- `--timestamp` an unique experiment string
- `--municipio` use which validation method, either `blockCV`, `bolivar` or ``
- `--subset` use which subset of features, single (distance to historical landine), geo (geospatial features) or full (all 70 features)
- `--model` which model to use, `TabCmpt` is the RELand model, other options can be `MLP`, `TabNet`, `LR`, `RF`, `SVM`, `LGBM`
- `--objective` using irm, erm, or pnorm
- `--n_step` number of decision blocks
- `--warm_start` directory with checkpoints

Your final result for this run will be stored under `/experiments/<timestamp>` that contains
- all current `.py` files in the root directory
- a `<municipality>.pth` model for each municipality
- an `<municipality>.png` image for each municipality that visualizes the ground truth and prediction
- `config.json` with current configuration (hyper)parameters
- `metrics.json` that contains all 4 metrics
- `predicted_proba.csv` that combines validation prediction for all municipalities
- `feature_importance.csv` with global feature importance if applicable. Black-box models generate -1 values for all features.

# Acknowledgment
We borrow and edit packages including [ood-bench](https://github.com/m-Just/OoD-Bench), [scikit-learn/tree](https://github.com/scikit-learn/scikit-learn/tree/9aaed498795f68e5956ea762fef9c440ca9eb239/sklearn/tree), [pytorch-tabnet](https://github.com/dreamquark-ai/tabnet).

# Citations
If you find our work helpful, please consider cite our paper:
```
@article{10.1145/3648437,
author = {Dulce Rubio, Mateo and Zeng, Siqi and Wang, Qi and Alvarado, Didier and Moreno Rivera, Francisco and Heidari, Hoda and Fang, Fei},
title = {RELand: Risk Estimation of Landmines via Interpretable Invariant Risk Minimization},
year = {2024},
issue_date = {June 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {2},
number = {2},
url = {https://doi.org/10.1145/3648437},
doi = {10.1145/3648437},
month = jun,
articleno = {23},
numpages = {29},
keywords = {Landmines, demining, risk assessment}
}
```
