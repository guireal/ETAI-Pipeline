# ETAI Predictive Pipeline - Two-Year Recidivism

This project predicts two-year recidivism using ProPublica's COMPAS dataset.

The pipeline includes data cleaning, preprocessing, model construction, cross-validation, hyperparameter tuning, evaluation and fairness analysis.

## Project structure

- `main.py`: Main pipeline entry point.
- `config.yaml`: Pipeline configuration.
- `requirements.txt`: Python dependencies.
- `src/data.py`: Data loading.
- `src/preprocessing.py`: Data cleaning and preprocessing.
- `src/model.py`: Model construction.
- `src/evaluate.py`: Model evaluation and fairness analysis.
- `src/tuning.py`: Optuna hyperparameter tuning and nested cross-validation.
- `src/results.py`: Experiment reports and scoreboard.
- `data/`: Dataset and data dictionary.
- `results/`: Generated experiment reports and scoreboard.

## Pipeline progress

| Week | Focus | Main additions |
|---|---|---|
| 2 | Baseline pipeline | Data loading, basic preprocessing, logistic regression and fairness check. |
| 3 | EDA and preprocessing | Missing-value analysis, domain checks, duplicate handling and improved preprocessing. |
| 4 | Cross-validation | Locked test set, stratified 5-fold CV and out-of-fold evaluation. |
| 5 | Hyperparameter tuning | Optuna and nested cross-validation. |
| 6 | Ensembles | Random Forest, Oblique Random Forest, AdaBoost and corrected paired testing. |
| 7 | Model evaluation | Gradient Boosting, HistGradientBoosting, Naive Bayes, multiple metrics, probabilities and experiment scoreboard. |

## Available models

- `dummy`
- `logistic_regression`
- `decision_tree`
- `random_forest`
- `oblique_random_forest`
- `adaboost`
- `gradient_boosting`
- `hist_gradient_boosting`
- `naive_bayes`

The selected model and its parameters are configured in `config.yaml`.

## Model evaluation

The pipeline uses stratified cross-validation on the development set. The final test set is kept separate and is not used during tuning or cross-validation.

The evaluation metrics include:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- F-beta score
- ROC-AUC
- Average precision (PR-AUC)

The selection metric is configured through `cv.selection_metric` and is used by Optuna and the experiment scoreboard.

Out-of-fold probabilities are used for the confusion matrix, classification report and fairness analysis.

The fairness analysis compares the model's false positive rate with the COMPAS score across racial groups.

## Hyperparameter tuning

Optuna is used to tune model hyperparameters.

When `tuning.enabled` is true:

1. Nested cross-validation evaluates the tuning procedure.
2. Each outer fold runs an Optuna study using inner cross-validation.
3. The selected model is evaluated on the outer validation fold.
4. A final tuning study is performed on the complete development set.
5. The final model is fitted on all development rows.

The locked test set is not used in this process.

## Experiment scoreboard

Each run saves:

- `results/runs.csv`: One row per experiment.
- `results/run_<timestamp>.txt`: Detailed report for each run.

The scoreboard records model settings, preprocessing configuration, tuned parameters, selection metric, mean score, standard deviation, untuned score, dummy baseline and fold scores.

The function `corrected_paired_test()` in `src/evaluate.py` can be used to compare two models statistically.

## Running the project

Activate the virtual environment in PowerShell:

```powershell
venv\Scripts\Activate.ps1

## Week 8 - XGBoost and Feature Selection

Week 8 introduces XGBoost and feature selection.

XGBoost is added as a new model using the xgboost Python package. It is now the default model in the pipeline.

Feature selection is implemented through the VoteSelector class. It uses three models to select relevant features:

- Random Forest
- Gradient Boosting
- L1 Logistic Regression

Each model votes for the features it considers important. Features with at least the configured number of votes are kept.

Feature selection is included inside the pipeline to avoid data leakage during cross-validation.

The number of required votes can also be tuned with Optuna.

The pipeline now caches preprocessing and feature selection steps during tuning to avoid repeating unnecessary work.
