"""Saving each run's results to disk."""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """Writes the full run report to a timestamped file."""
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params', {})}\n"
        f"Test size: {config['test_set']['test_size']}  "
        f"random_state: {config['test_set']['random_state']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path
