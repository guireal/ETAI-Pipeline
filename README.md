# ETAI Predictive Pipeline

This project develops a machine learning pipeline for classification problems. It started with the COMPAS two-year recidivism dataset and was extended to work with the Adult income dataset.

The pipeline includes data loading, cleaning, preprocessing, feature selection, model construction, cross-validation, hyperparameter tuning, evaluation, fairness analysis and prediction of new observations.

## Project structure

```text
.
├── main.py
├── predict.py
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
│   ├── compas_new_people.csv
│   ├── adult_train.csv
│   ├── adult_new.csv
│   ├── README.md
│   └── ADULT_README.md
├── models/
├── results/
└── results_adult/
    └── adult_predictions.csv
```

The `models/` directory is generated locally and is not tracked by Git.

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
| 10 | Missing values and deployment | Five numerical imputation strategies, scaling before imputation, saving fitted models and standalone prediction of new observations. |

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

## Week 10 — Missing Values and Deployment

### Numerical imputation

The numerical preprocessing pipeline supports five imputation strategies:

| Strategy | Description |
|---|---|
| `median` | Replaces missing values with the training-set median. |
| `mean` | Replaces missing values with the training-set mean. |
| `knn` | Uses the nearest observations to estimate missing values. |
| `iterative` | Uses other numerical features to estimate missing values through iterative imputation. |
| `random_forest` | Uses an iterative imputer with a Random Forest estimator. |

The strategy is selected through `preprocessing.imputation.numeric_strategy` in the configuration file.

Numerical features are scaled before imputation. This is particularly important for KNN because distance calculations should not be dominated by features with larger numerical scales.

The imputation parameters can also be configured:

- `n_neighbors`: number of neighbours used by KNN.
- `max_iter`: maximum number of iterations for iterative imputation.
- `n_estimators`: number of trees used by the Random Forest imputer.

Categorical features continue to use the configured categorical imputation strategy.

All imputers are fitted inside the preprocessing pipeline, using only the training data of each cross-validation fold. The same fitted preprocessing steps are then used for new observations.

### Saving the fitted model

After training, the final fitted pipeline is saved with Joblib.

The pipeline includes preprocessing, feature selection when enabled, and the trained model.

The model paths are configured separately:

- COMPAS: `models/model.joblib`
- Adult: `models/model_adult.joblib`

The `models/` directory is excluded from Git because fitted model files are generated locally.

### Predicting new observations

The `predict.py` script loads a previously saved model and generates predictions without retraining.

For COMPAS:

```powershell
python predict.py data/compas_new_people.csv predictions.csv
```

For Adult:

```powershell
python predict.py data/adult_new.csv results_adult/predictions.csv config_adult.yaml
```

The prediction script accepts an input CSV, an optional output path and an optional configuration file.

Every input observation receives one prediction. The output contains:

- `id`: original observation identifier.
- `probability`: predicted probability of the positive class.
- `prediction`: binary prediction using a threshold of 0.5.

New observations are cleaned using the same configured rules as the training data. Duplicate rows are not removed, so every input observation is preserved.

### Week 10 testing

The following tests were performed:

- All five numerical imputation strategies were tested on COMPAS and Adult data.
- Scaling before imputation was verified.
- Missing and invalid transformed values were checked.
- Both COMPAS and Adult training workflows successfully saved fitted models.
- The saved models were loaded independently through `predict.py`.
- COMPAS standalone prediction generated 40 predictions for 40 input observations.
- Adult standalone prediction generated 2,000 predictions for 2,000 input observations.
- Prediction IDs, probabilities and binary labels were validated.

The small training samples used for the integration tests were only intended to verify the workflow. Their evaluation metrics are not reported as final model performance.