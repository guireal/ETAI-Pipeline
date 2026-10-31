"""Saving experiment results to disk."""

import os
import pandas as pd
from datetime import datetime


def save_scores(path, row):
    """Append one experiment to the results scoreboard."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        scores = pd.read_csv(path)
        scores = pd.concat(
            [scores, pd.DataFrame([row])],
            ignore_index=True
        )
    else:
        scores = pd.DataFrame([row])

    scores.to_csv(path, index=False)

    return scores


def save_run(results_dir, run, config, report_text):
    """Save the configuration summary and report to a text file."""
    os.makedirs(results_dir, exist_ok=True)

    path = os.path.join(results_dir, f"run_{run}.txt")

    tuning = config.get("tuning", {})

    if tuning.get("enabled", False):
        tuning_status = (
            f"enabled ({tuning.get('n_trials')} trials, "
            f"{tuning.get('n_splits')} folds)"
        )
    else:
        tuning_status = "disabled"

    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']} "
        f"params={config['model'].get('params', {})}\n"
        f"Preprocessing: "
        f"encoder={config['preprocessing'].get('encoder')} "
        f"scaler={config['preprocessing'].get('scaler')}\n"
        f"CV: {config['cv']['n_splits']} folds\n"
        f"Metrics: {config['cv'].get('metrics', [])}\n"
        f"Selection metric: "
        f"{config['cv'].get('selection_metric', 'roc_auc')}\n"
        f"Tuning: {tuning_status}\n"
        f"Test size: {config['test_set']['size']} "
        f"random_state={config['test_set']['random_state']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write(header + report_text)

    return path