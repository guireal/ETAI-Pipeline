import argparse
import os

import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, split_features_target


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def predict(df, pipeline, config):
    """Generate one prediction for every row in the input dataset."""

    id_column = config["data"].get(
        "id_column",
        config.get("diagnostics", {}).get("id_column", "id")
    )

    if id_column not in df.columns:
        raise ValueError(
            f"Identifier column '{id_column}' was not found."
        )

    required_columns = (
        config["preprocessing"].get("numeric_features", [])
        + config["preprocessing"].get("categorical_features", [])
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {missing_columns}"
        )

    original_ids = df[id_column].copy()

    clean = clean_dataset(
        df,
        config.get("diagnostics", {})
    )

    X, _, _ = split_features_target(
        clean,
        config["data"],
        config["preprocessing"].get("mnar_indicator_sources", [])
    )

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return pd.DataFrame({
        id_column: original_ids,
        "probability": probabilities,
        "prediction": predictions
    })


def main():
    parser = argparse.ArgumentParser(
        description="Predict new observations using a saved model."
    )

    parser.add_argument(
        "input",
        help="CSV file containing the observations to predict."
    )

    parser.add_argument(
        "output",
        nargs="?",
        default="predictions.csv",
        help="Output CSV file."
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Configuration file used to train the model."
    )

    args = parser.parse_args()

    config = load_config(args.config)

    model_path = config.get("output", {}).get(
        "model_path",
        "models/model.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model not found: {model_path}. "
            "Run main.py with this configuration first."
        )

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    pipeline = joblib.load(model_path)

    df = pd.read_csv(args.input)

    predictions = predict(
        df,
        pipeline,
        config
    )

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    predictions.to_csv(
        args.output,
        index=False
    )

    print(f"Model: {model_path}")
    print(f"Input rows: {len(df)}")
    print(f"Predictions generated: {len(predictions)}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
