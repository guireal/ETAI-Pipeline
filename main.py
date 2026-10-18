"""Entry point: run the complete prediction pipeline."""

import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    build_preprocessor,
    split_dev_test
)
from src.model import build_model, build_pipeline
from src.evaluate import (
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
from src.results import save_run


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

    # Separate features, target and fairness columns.
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

    preprocessor = build_preprocessor(
        config.get("preprocessing", {})
    )

    model = build_model(config["model"])
    pipeline = build_pipeline(preprocessor, model)

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=(
            cv_config.get("random_state")
            if shuffle else None
        )
    )

    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    # Evaluate the configured model using cross-validation.
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline,
        X_dev,
        y_dev,
        cv,
        scoring,
        n_jobs=n_jobs
    )

    report = (
        "Hyperparameters from config.yaml: "
        + str(config["model"].get("params", {}))
    )
    report += "\n" + cv_report(fold_scores, scoring)

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    study = None

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

        nested_scores, y_oof = nested_cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv,
            inner_cv,
            scoring,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs
        )

        report += "\n\nNested cross-validation:\n"
        report += cv_report(nested_scores, scoring)

        # Tune the final model using all development data.
        pipeline, study = tune_pipeline(
            pipeline,
            X_dev,
            y_dev,
            inner_cv,
            scoring,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs
        )

        report += "\n\n" + tuning_report(
            study,
            nested_scores,
            scoring
        )

    report += "\n\n" + oof_classification_report(
        y_dev,
        y_oof
    )

    report += "\n" + fairness_report(
        y_dev,
        y_oof,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"]
    )

    # Fit the selected pipeline on all development data.
    pipeline.fit(X_dev, y_dev)

    refit_message = (
        f"Final model: {config['model']['type']} fitted on "
        f"{len(X_dev)} development rows."
    )

    if study is not None:
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

    saved_path = save_run(
        results_dir,
        config,
        report
    )

    print(f"Full results saved to {saved_path}")


if __name__ == "__main__":
    main()