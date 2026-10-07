"""Entry point: run the whole pipeline."""
import yaml
from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, build_preprocessor, split_dev_test
from src.model import build_model, build_pipeline
from src.evaluate import cross_validate_pipeline, evaluate_oof, fairness_report_oof
from src.results import save_run


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_df = load_data(config["data"]["path"])
    cleaned_df = clean_dataset(raw_df, config.get("diagnostics", {}))

    id_col = config.get("diagnostics", {}).get("id_column", "id")
    deduped_df = drop_duplicate_rows(cleaned_df, id_col=id_col)

    X_dev, y_dev, extras_dev, X_test, y_test, extras_test = split_dev_test(
        deduped_df,
        target=config["data"]["target"],
        sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        mnar_sources=config.get("preprocessing", {}).get("mnar_indicator_sources", []),
        test_size=config["test_set"]["test_size"],
        random_state=config["test_set"]["random_state"],
    )

    preprocessor = build_preprocessor(config.get("preprocessing", {}))
    model = build_model(config["model"])
    pipeline = build_pipeline(preprocessor, model)

    oof_preds, cv_report = cross_validate_pipeline(
        pipeline,
        X_dev,
        y_dev,
        n_splits=config["cv"]["n_splits"],
        random_state=config["cv"]["random_state"],
    )

    clf_report = evaluate_oof(y_dev, oof_preds)
    fair_report = fairness_report_oof(
        y_dev,
        oof_preds,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    pipeline.fit(X_dev, y_dev)

    results_dir = config.get("output", {}).get("results_dir", "results")
    full_report = f"{cv_report}\n\n{clf_report}\n\n{fair_report}"
    saved_path = save_run(results_dir, config, full_report)
    print(f"Full results saved to {saved_path}")


if __name__ == "__main__":
    main()