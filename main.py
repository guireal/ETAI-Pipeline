
import os
import sys
import shutil
import tempfile
from datetime import datetime

import optuna
import pandas as pd
import yaml
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
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_run, save_scores


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path="config.yaml"):
    config = load_config(config_path)
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    print(f"Configuration: {config_path}")

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config.get("diagnostics", {}))

    df_clean = drop_duplicate_rows(
        df_clean,
        config.get("diagnostics", {}).get("id_column")
    )

    mnar_sources = config["preprocessing"].get(
        "mnar_indicator_sources", []
    )

    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        mnar_sources
    )

    if y is None:
        raise ValueError("The target column was not found in the training data.")

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"]
    )

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None
    )

    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)

    selection_config = config.get("feature_selection") or {}
    selection_enabled = selection_config.get("enabled", False)

    print(f"Model: {model_type}")
    print(f"Selection metric: {selection_metric}")
    print(f"Feature selection: {selection_enabled}")
    print(f"Development rows: {len(X_dev)}")
    print(f"Locked test rows: {len(X_test)}")

    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")

    try:
        pipeline = build_pipeline(
            config["preprocessing"],
            config["model"],
            selection_config,
            memory=cache_dir
        )

        floor_pipeline = build_pipeline(
            config["preprocessing"],
            {"type": "dummy"}
        )

        floor_scores, _ = cross_validate_pipeline(
            floor_pipeline,
            X_dev,
            y_dev,
            cv,
            {selection_metric: scorers[selection_metric]},
            selection_metric,
            n_jobs
        )

        fold_scores, oof_proba = cross_validate_pipeline(
            pipeline,
            X_dev,
            y_dev,
            cv,
            scorers,
            selection_metric,
            n_jobs
        )

        floor = floor_scores["validation"].mean()
        untuned = fold_scores["validation"].mean()

        print("\nDefault hyperparameters")
        print(f"{selection_metric}: {untuned:.3f}")

        report = (
            f"Configuration: {config_path}\n"
            f"Hyperparameters: {config['model'].get('params')}\n\n"
            + cv_report(
                fold_scores,
                metrics,
                selection_metric,
                "Cross-validation"
            )
            + f"\n\nNo-model floor: {floor:.3f}"
        )

        final_scores = fold_scores
        tuning_config = config.get("tuning", {})
        tuning_enabled = tuning_config.get("enabled", False)
        study = None

        if tuning_enabled:
            search_spaces = tuning_config.get("search_spaces") or {}

            if model_type not in search_spaces:
                raise ValueError(
                    f"No tuning search space for model '{model_type}'."
                )

            search_space = dict(search_spaces[model_type])

            if selection_enabled:
                search_space.update(
                    selection_config.get("search_space") or {}
                )

            n_trials = tuning_config["n_trials"]
            tuning_seed = tuning_config["random_state"]

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            inner_cv = StratifiedKFold(
                n_splits=tuning_config["n_splits"],
                shuffle=True,
                random_state=tuning_seed
            )

            print("\nRunning nested cross-validation...")

            final_scores, oof_proba = nested_cross_validate(
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

            print(
                f"Nested CV {selection_metric}: "
                f"{final_scores['validation'].mean():.3f}"
            )

            report += "\n\n" + cv_report(
                final_scores,
                metrics,
                selection_metric,
                "Nested cross-validation"
            )

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
                final_scores,
                selection_metric
            )

        print("\nAll metrics")

        for name in metrics:
            print(
                f"{name}: {final_scores[name].mean():.3f} "
                f"+/- {final_scores[name].std(ddof=1):.3f}"
            )

        report += "\n\n" + oof_classification_report(
            y_dev, oof_proba
        )

        report += "\n" + fairness_report(
            y_dev,
            oof_proba,
            extras_dev,
            sensitive_attr=config["data"].get("sensitive_attr")
        )

        # Fit the final model on all development observations.
        pipeline.fit(X_dev, y_dev)

        refit = (
            f"Final model: {model_type} refit on "
            f"{len(X_dev)} development rows."
        )

        if selection_enabled:
            all_features = pipeline.named_steps["prep"].get_feature_names_out()
            kept = pipeline.named_steps["select"].get_feature_names_out(
                all_features
            ).tolist()

            refit += (
                f"\nFeatures kept: {len(kept)} of {len(all_features)}: "
                + ", ".join(kept)
            )

        report += "\n" + refit + "\n"

        # Predict a separate new dataset when one is configured.
        new_data_path = config["data"].get("new_data_path")
        predictions_path = config["data"].get("predictions_path")

        if new_data_path:
            new_df = load_data(new_data_path)
            id_column = config["data"].get("id_column", "id")

            if id_column not in new_df.columns:
                raise ValueError(
                    f"Identifier column '{id_column}' "
                    f"was not found in {new_data_path}."
                )

            new_ids = new_df[id_column].copy()

            # Clean the new data but do not remove duplicate rows.
            new_clean = clean_dataset(
                new_df,
                config.get("diagnostics", {})
            )

            X_new, _, _ = split_features_target(
                new_clean,
                config["data"],
                mnar_sources
            )

            probabilities = pipeline.predict_proba(X_new)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)

            prediction_df = pd.DataFrame({
                id_column: new_ids,
                "probability": probabilities,
                "prediction": predictions
            })

            if predictions_path:
                os.makedirs(
                    os.path.dirname(predictions_path) or ".",
                    exist_ok=True
                )

                prediction_df.to_csv(
                    predictions_path,
                    index=False
                )

                prediction_message = (
                    f"Predictions saved to {predictions_path}. "
                    f"Rows: {len(prediction_df)}."
                )

                print(prediction_message)
                report += "\n" + prediction_message + "\n"

        row = {
            "run": run,
            "data": config["data"]["path"],
            "model": model_type,
            "tuned": tuning_enabled,
            "encoder": config["preprocessing"]["encoder"],
            "scaler": config["preprocessing"]["scaler"],
            "feature_selection": (
                f"min_votes {pipeline.named_steps['select'].min_votes}"
                if selection_enabled else "off"
            ),
            "params": str(config["model"].get("params")),
            "tuned_params": str(study.best_params) if tuning_enabled else "",
            "metric": selection_metric,
            "mean": final_scores["validation"].mean(),
            "std": final_scores["validation"].std(ddof=1),
            "untuned": untuned,
            "floor": floor,
            **{
                f"fold_{i}": score
                for i, score in enumerate(
                    final_scores["validation"], start=1
                )
            }
        }

        runs = save_scores(
            config.get("output", {}).get(
                "scoreboard", "results/runs.csv"
            ),
            row
        )

        same = runs[
            (runs["data"] == row["data"])
            & (runs["metric"] == selection_metric)
        ]

        board = (
            same.sort_values("mean", ascending=False)
            [["run", "model", "tuned", "feature_selection",
              "mean", "std", "untuned", "floor"]]
            .rename(columns={"mean": selection_metric})
            .to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}"
            )
        )

        print(f"\nScoreboard\n{board}")
        report += f"\nScoreboard:\n{board}\n"

        path = save_run(
            config.get("output", {}).get(
                "results_dir", "results"
            ),
            run,
            config,
            report
        )

        print(f"\n{refit}")
        print(f"Full report saved to: {path}")

    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    selected_config = (
        sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    )
    main(selected_config)
