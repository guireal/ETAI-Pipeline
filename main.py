"""Entry point: run the complete prediction pipeline."""

import optuna
import yaml
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    split_dev_test
)
from src.model import build_pipeline
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report
)
from src.tuning import (
    tune_pipeline,
    nested_cross_validate,
    tuning_report
)
from src.results import save_run, save_scores


def main():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_df = load_data(config["data"]["path"])

    cleaned_df = clean_dataset(
        raw_df,
        config.get("diagnostics", {})
    )

    id_col = config.get("diagnostics", {}).get(
        "id_column", "id"
    )

    deduped_df = drop_duplicate_rows(
        cleaned_df,
        id_column=id_col
    )

    mnar_sources = config.get(
        "preprocessing", {}
    ).get("mnar_indicator_sources", [])

    X, y, extras = split_features_target(
        deduped_df,
        config["data"],
        mnar_sources
    )

    # Keep the final test set separate from model selection.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = (
        split_dev_test(
            X,
            y,
            extras,
            test_size=config["test_set"]["size"],
            random_state=config["test_set"]["random_state"]
        )
    )

    pipeline = build_pipeline(
        config["preprocessing"],
        config["model"]
    )

    cv_config = config["cv"]

    metrics = cv_config["metrics"]
    selection_metric = cv_config.get(
        "selection_metric", "roc_auc"
    )

    scorers = build_scorers(metrics)

    if selection_metric not in scorers:
        raise ValueError(
            f"Selection metric '{selection_metric}' "
            f"is not included in the configured metrics."
        )

    n_jobs = cv_config.get("n_jobs", 1)
    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=(
            cv_config.get("random_state")
            if shuffle else None
        )
    )

    # Evaluate a dummy classifier as a baseline.
    floor_pipeline = build_pipeline(
        config["preprocessing"],
        {"type": "dummy", "params": {}}
    )

    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline,
        X_dev,
        y_dev,
        cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric,
        n_jobs=n_jobs
    )

    floor = floor_scores["validation"].mean()

    # Evaluate the configured model without tuning.
    fold_scores, oof_proba = cross_validate_pipeline(
        pipeline,
        X_dev,
        y_dev,
        cv,
        scorers,
        selection_metric,
        n_jobs=n_jobs
    )

    untuned = fold_scores[selection_metric].mean()

    report = (
        "Hyperparameters from config.yaml: "
        + str(config["model"].get("params", {}))
    )

    report += (
        f"\n\nDummy baseline ({selection_metric}): "
        f"{floor:.3f}"
    )

    report += (
        f"\nUntuned model ({selection_metric}): "
        f"{untuned:.3f}"
    )

    report += "\n\n" + cv_report(
        fold_scores,
        metrics,
        selection_metric,
        "Untuned model cross-validation"
    )

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)

    study = None
    final_scores = fold_scores

    if tuning_enabled:
        model_type = config["model"]["type"]

        search_spaces = tuning_config.get(
            "search_spaces", {}
        )

        if model_type not in search_spaces:
            raise ValueError(
                f"No tuning search space is configured "
                f"for '{model_type}'."
            )

        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]

        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed
        )

        optuna.logging.set_verbosity(
            optuna.logging.WARNING
        )

        print(
            f"\nNested CV: {cv.get_n_splits()} outer folds, "
            f"{n_trials} trials per outer fold, "
            f"{inner_cv.get_n_splits()} inner folds."
        )

        nested_scores, oof_proba = nested_cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv,
            inner_cv,
            scorers,
            selection_metric,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs
        )

        final_scores = nested_scores

        report += "\n\n" + cv_report(
            nested_scores,
            metrics,
            selection_metric,
            "Nested cross-validation"
        )

        # Tune the final model using all development data.
        pipeline, study = tune_pipeline(
            pipeline,
            X_dev,
            y_dev,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs
        )

        report += "\n\n" + tuning_report(
            study,
            nested_scores,
            selection_metric
        )

    # Classification and fairness reports use OOF probabilities.
    report += "\n\n" + oof_classification_report(
        y_dev,
        oof_proba
    )

    report += "\n" + fairness_report(
        y_dev,
        oof_proba,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"]
    )

    # Fit the selected pipeline on all development data.
    pipeline.fit(X_dev, y_dev)

    refit_message = (
        f"Final model: {config['model']['type']} fitted on "
        f"{len(X_dev)} development rows."
    )

    tuned_params = {}

    if study is not None:
        tuned_params = study.best_params
        refit_message += (
            f" Tuned parameters: {study.best_params}"
        )

    print(refit_message)
    report += "\n" + refit_message + "\n"

    # The test set remains untouched during model selection.
    locked_message = (
        f"Locked test set: {len(X_test)} rows set aside "
        f"and not evaluated. Development set: "
        f"{len(X_dev)} rows."
    )

    print(locked_message)
    report += "\n" + locked_message + "\n"

    results_dir = config.get("output", {}).get(
        "results_dir", "results"
    )

    run = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_path = save_run(
        results_dir,
        run,
        config,
        report
    )

    # Save the experiment to the scoreboard.
    scoreboard_path = config.get(
        "output", {}
    ).get(
        "scoreboard", "results/runs.csv"
    )

    row = {
        "run": run,
        "data": config["data"]["path"],
        "model": config["model"]["type"],
        "tuned": tuning_enabled,
        "encoder": config["preprocessing"].get("encoder"),
        "scaler": config["preprocessing"].get("scaler"),
        "params": str(config["model"].get("params", {})),
        "tuned_params": str(tuned_params),
        "metric": selection_metric,
        "mean": final_scores[selection_metric].mean(),
        "std": final_scores[selection_metric].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
        "fold_scores": str(
            final_scores[selection_metric].tolist()
        )
    }

    scoreboard = save_scores(scoreboard_path, row)

    comparison = scoreboard[
        (scoreboard["data"] == config["data"]["path"])
        & (scoreboard["metric"] == selection_metric)
    ].sort_values(
        "mean",
        ascending=False
    )

    print("\nExperiment scoreboard:")
    print(comparison.to_string(index=False))

    print(f"\nFull results saved to {saved_path}")
    print(f"Scoreboard saved to {scoreboard_path}")


if __name__ == "__main__":
    main()