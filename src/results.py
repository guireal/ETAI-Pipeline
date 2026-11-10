import os
import pandas as pd


def save_scores(path, row):
    runs = (
        pd.read_csv(path, dtype={"run": str})
        if os.path.exists(path)
        else pd.DataFrame()
    )

    runs = pd.concat(
        [runs, pd.DataFrame([row])],
        ignore_index=True
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)

    return runs


def save_run(results_dir, run, config, report_text):
    os.makedirs(results_dir, exist_ok=True)

    path = os.path.join(results_dir, f"run_{run}.txt")

    cv = config["cv"]
    tuning = config.get("tuning", {})
    selection = config.get("feature_selection") or {}

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
        f"CV: {cv['n_splits']} stratified folds\n"
        f"Metrics: {cv['metrics']}\n"
        f"Selection metric: {cv['selection_metric']}\n"
        f"Feature selection: {selection_line}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={config['test_set']['size']} "
        f"(not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as file:
        file.write(header + report_text)

    return path