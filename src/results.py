"""Saving each run's results to disk."""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """Save the configuration summary and report to a text file."""
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    tuning = config.get("tuning", {})

    if tuning.get("enabled", False):
        tuning_status = (
            f"enabled ({tuning.get('n_trials')} trials, "
            f"{tuning.get('n_splits')} folds)"
        )
    else:
        tuning_status = "disabled"

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']} "
        f"params={config['model'].get('params', {})}\n"
        f"Preprocessing: "
        f"encoder={config['preprocessing'].get('encoder')} "
        f"scaler={config['preprocessing'].get('scaler')}\n"
        f"CV: {config['cv']['n_splits']} folds "
        f"metric={config['cv'].get('scoring', 'accuracy')}\n"
        f"Tuning: {tuning_status}\n"
        f"Test size: {config['test_set']['size']} "
        f"random_state={config['test_set']['random_state']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write(header + report_text)

    return path