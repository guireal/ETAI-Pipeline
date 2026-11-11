
# ETAI Predictive Pipeline

This project develops a machine learning pipeline for classification problems. It started with the COMPAS two-year recidivism dataset and was extended to work with the Adult income dataset.

The pipeline includes data loading, cleaning, preprocessing, feature selection, model construction, cross-validation, hyperparameter tuning, evaluation and fairness analysis.

## Project structure

```text
.
├── main.py
├── config.yaml
├── config_adult.yaml
├── requirements.txt
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   ├── tuning.py
│   └── results.py
├── data/
│   ├── compas_two_year_recidivism.csv
│   ├── adult_train.csv
│   ├── adult_new.csv
│   └── ADULT_README.md
├── results/
└── results_adult/
    └── adult_predictions.csv
```

## Pipeline progress

| Week | Focus | Main additions |
|---|---|---|
| 2 | Baseline pipeline | Data loading, basic preprocessing, logistic regression and fairness check. |
| 3 | EDA and preprocessing | Missing-value analysis, domain checks, duplicate handling and improved preprocessing. |
| 4 | Cross-validation | Locked test set, stratified 5-fold CV and out-of-fold evaluation. |
| 5 | Hyperparameter tuning | Optuna and nested cross-validation. |
| 6 | Ensembles | Random Forest, Oblique Random Forest, AdaBoost and corrected paired testing. |
| 7 | Model evaluation | Gradient Boosting, HistGradientBoosting, Naive Bayes, multiple metrics, probabilities and experiment scoreboard. |
| 8 | XGBoost and feature selection | XGBoost, VoteSelector and joint tuning of model parameters and the feature-selection threshold. |
| 9 | Beyond COMPAS | Configuration-based execution, generalised cleaning and fairness evaluation, Adult income predictions for new observations. |

## Available models

- Dummy classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Oblique Random Forest
- AdaBoost
- Gradient Boosting
- Histogram Gradient Boosting
- Naive Bayes
- XGBoost

The selected model and its parameters are configured in the YAML files.

## Running the project

Activate the virtual environment in PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Run the original COMPAS pipeline:

```powershell
python main.py
```

Run the Adult income pipeline:

```powershell
python main.py config_adult.yaml
```

The configuration file determines the dataset, target, features, preprocessing, model, evaluation settings and output folder.

## Model evaluation

The pipeline uses stratified cross-validation on the development set. The final test set is kept separate and is not used during model selection or tuning.

The reported metrics include:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- F-beta score
- ROC-AUC
- Average precision

The selection metric is configured through `cv.selection_metric` and is used by Optuna and the experiment scoreboard.

Out-of-fold probabilities are used for the classification report and fairness analysis.

## Hyperparameter tuning

Optuna is used to tune model hyperparameters.

When `tuning.enabled` is true:

1. Nested cross-validation evaluates the tuning procedure.
2. Each outer fold runs an Optuna study using inner cross-validation.
3. The selected model is evaluated on an outer validation fold.
4. A final tuning study is performed on the complete development set.
5. The final model is fitted on all development rows.

The locked test set is not used during this process.

## Feature selection

The `VoteSelector` class uses three models to select relevant features:

- Random Forest
- Gradient Boosting
- L1 Logistic Regression

Each model votes for the features it considers important. Features with at least the configured number of votes are kept.

Feature selection is inside the pipeline, so it is fitted separately on the training rows of each cross-validation fold.

## Beyond COMPAS — Week 9, Option A

### Objective

The Adult income dataset changes the data but keeps the binary classification problem.

The objective is to predict whether a person earns more than 50,000 USD per year.

The training dataset contains 6,000 observations. The new dataset contains 2,000 observations without income labels.

### Running the Adult pipeline

```powershell
python main.py config_adult.yaml
```

The Adult configuration has its own data settings, preprocessing, model, tuning settings and output folder. The original COMPAS configuration remains available through `python main.py`.

### Data decisions

| Column or group | Decision |
|---|---|
| `income_over_50k` | Binary target: 1 means income above 50K. |
| `id` | Identifier only; preserved for the prediction output and never used as a feature. |
| `fnlwgt` | Excluded because it is a census sampling weight, not a characteristic of the person. |
| `education` | Excluded because `education_num` contains the same information in ordered form. |
| `sex` | Selected as the sensitive attribute for fairness evaluation and excluded from model features. |
| `race` | Also excluded from model features so that neither protected attribute is used to predict income. |
| `v01`–`v25` | Kept as candidate features; feature selection decides whether they are useful. |

### What broke, why and how it was fixed

| What broke | Why | Fix |
|---|---|---|
| The main program only used `config.yaml`. | The configuration path was fixed in the code. | `main.py` now accepts a configuration filename and still defaults to `config.yaml`. |
| Adult missing values written as `?` were not recognised. | The original cleaning handled placeholders only in columns with configured category mappings. | Cleaning now also uses the configured `categorical_columns` list. |
| The fairness report expected `score_text`. | That column belongs to COMPAS and does not exist in Adult. | Fairness evaluation now works with the configured sensitive attribute and only compares against COMPAS when its score column exists. |
| The new Adult file has no target column. | The original training workflow expected labelled observations. | The final fitted pipeline can predict a new file without requiring its target. |
| New observations must all receive predictions. | Training duplicate removal must not remove prediction rows. | New data is cleaned and transformed without dropping duplicate rows. |

### Adult preprocessing decisions

The three `?` placeholders are converted into missing values:

- `workclass`: 379 missing values.
- `occupation`: 381 missing values.
- `native_country`: 86 missing values.

Missingness indicators are used for `workclass` and `occupation`, since their missingness carries information about income. Other missing values are imputed without additional indicators.

The value `99999` in `capital_gain` is retained because it is a valid census top code, not an invalid value.

The age validity rule starts at 17 because 17-year-olds are valid observations in this dataset.

The original dataset contains no repeated IDs or duplicate rows.

### Experiment results

The professor's target scores are ROC-AUC at least 0.90, accuracy at least 0.855 and F1 at least 0.66.

| Run | ROC-AUC | Accuracy | F1 | Tuning |
|---|---:|---:|---:|---|
| XGBoost, default parameters | 0.913 | 0.864 | 0.686 | Off |
| XGBoost, nested CV | 0.910 | 0.860 | 0.674 | On |
| Required target | 0.900 | 0.855 | 0.660 | — |

Both runs reached all three targets. The final tuned run was retained because it evaluates the tuning procedure using nested cross-validation.

The corrected paired test comparing the two runs gave a mean ROC-AUC difference of 0.0036 in favour of the untuned model, with a p-value of 0.141. The difference was not statistically significant at the 5% level.

### Features selected

The final tuned model kept 8 of the 37 candidate features:

- `age`
- `education_num`
- `capital_gain`
- `capital_loss`
- `hours_per_week`
- `marital_status`
- `occupation`
- `relationship`

None of the anonymised `v01`–`v25` features was selected.

### Predictions for new observations

The final model generated predictions for all 2,000 observations in `adult_new.csv`.

The output is saved in:

`results_adult/adult_predictions.csv`

It contains three columns:

- `id`
- `probability`
- `prediction`

The original IDs and row count were verified. All probabilities were valid and predictions were binary.

### Reflection

The hardest assumptions to identify were the ones tied directly to COMPAS, especially the cleaning of placeholder values and the fairness report's dependence on `score_text`. Making these steps configurable allowed the same pipeline to work with Adult without changing the original COMPAS configuration. XGBoost, sensible preprocessing and feature selection allowed the Adult model to reach all three target scores. The untuned model performed slightly better, but the tuned model also met every target.
