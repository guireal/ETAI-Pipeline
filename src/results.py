"""Saving experiment results and reports."""

import os
import pandas as pd


def save_scores(path, row):
    """Append one experiment to the scoreboard."""

    runs = (
        pd.read_csv(path, dtype={"run": str})
        if os.path.exists(path)
        else pd.DataFrame()
    )

    runs = pd.concat(
        [runs, pd.DataFrame([row])],
        ignore_index=True
    )

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True
    )

    runs.to_csv(path, index=False)

    return runs


def save_run(results_dir, run, config, report_text):
    """Save the full report for an experiment."""

    os.makedirs(results_dir, exist_ok=True)

    path = os.path.join(
        results_dir,
        f"run_{run}.txt"
    )

    cv = config["cv"]
    tuning = config.get("tuning", {})
    selection = config.get("feature_selection") or {}
    imputation = config["preprocessing"].get("imputation", {})
    anomaly = config.get("anomaly") or {}

    if tuning.get("enabled"):
        tuning_line = (
            f"Optuna, {tuning['n_trials']} trials x "
            f"{tuning['n_splits']} folds, nested CV"
        )
    else:
        tuning_line = "off"

    if selection.get("enabled"):
        selection_line = (
            f"VoteSelector, min_votes={selection.get('min_votes', 2)}"
        )
    else:
        selection_line = "off"

    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']} "
        f"params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']} "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"Numeric imputer: "
        f"{imputation.get('numeric_strategy', 'median')}\n"
        f"Categorical imputer: "
        f"{imputation.get('categorical_strategy', 'most_frequent')}\n"
        f"Imputer parameters: "
        f"n_neighbors={imputation.get('n_neighbors', 5)}, "
        f"max_iter={imputation.get('max_iter', 10)}, "
        f"n_estimators={imputation.get('n_estimators', 50)}\n"
        f"Anomaly score: {anomaly.get('method', 'none')}\n"
        f"Anomaly n_neighbors: {anomaly.get('n_neighbors', 20)}\n"
        f"CV: {cv['n_splits']} stratified folds\n"
        f"Metrics: {cv['metrics']}\n"
        f"Selection metric: {cv['selection_metric']}\n"
        f"Feature selection: {selection_line}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={config['test_set']['size']} "
        f"(not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write(header + report_text)

    return path