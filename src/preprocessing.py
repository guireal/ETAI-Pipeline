"""Cleaning, feature splitting and leak-safe preprocessing."""
import numpy as np
import pandas as pd

from category_encoders import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)


def flag_invalid_values(df, rules):
    """Replace values outside configured domain rules with NaN."""
    report_rows = []

    for column, bounds in rules.items():
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(
            df[column], errors="coerce"
        )

        lower_ok = (
            numeric >= bounds["min"]
            if "min" in bounds
            else pd.Series(True, index=numeric.index)
        )
        upper_ok = (
            numeric <= bounds["max"]
            if "max" in bounds
            else pd.Series(True, index=numeric.index)
        )

        violations = numeric.notna() & ~(lower_ok & upper_ok)

        report_rows.append({
            "column": column,
            "rule": bounds,
            "violations": int(violations.sum())
        })

        df.loc[violations, column] = np.nan

    return pd.DataFrame(report_rows)


def _canonicalize_categories(
    df, columns_and_maps, placeholder_tokens
):
    out = df.copy()

    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue

        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()

        out[col] = lowered.map(mapping).fillna(cleaned)

        out.loc[
            out[col].astype(str).str.strip().isin(
                placeholder_tokens
            ),
            col
        ] = np.nan

    return out


def clean_dataset(df, diagnostics_config):
    """Clean categories, invalid values and redundant columns."""
    out = df.copy()

    placeholder_tokens = set(
        diagnostics_config.get("placeholder_tokens", [])
    )

    for col in diagnostics_config.get(
        "numeric_text_columns", []
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].replace(
                    list(placeholder_tokens), np.nan
                ),
                errors="coerce"
            )

    flag_invalid_values(
        out,
        diagnostics_config.get("validity_rules", {})
    )

    out = _canonicalize_categories(
        out,
        diagnostics_config.get("canonical_categories", {}),
        placeholder_tokens
    )

    columns_to_drop = [
        c for c in diagnostics_config.get(
            "redundant_columns", []
        )
        if c in out.columns
    ]

    return out.drop(columns=columns_to_drop)


def drop_duplicate_rows(df, id_column=None):
    """Drop duplicate records before splitting development and test data."""
    out = df.drop_duplicates()

    if id_column and id_column in out.columns:
        out = out.drop_duplicates(
            subset=id_column, keep="first"
        )

    return out


def add_missingness_indicators(df, mnar_indicator_sources):
    """Add flags before missing values are imputed."""
    out = df.copy()

    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = (
                out[col].isna().astype(int)
            )

    return out


def split_features_target(
    df, data_config, mnar_indicator_sources
):
    """Separate features, target and columns reserved for fairness auditing."""
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(
        df, mnar_indicator_sources
    )

    y = df[target] if target in df.columns else None

    extras_cols = [
        c for c in [sensitive_attr, "score_text"]
        if c in df.columns
    ]

    extras = (
        df[extras_cols].copy()
        if extras_cols else None
    )

    always_drop = set(drop_columns) | {target, sensitive_attr}

    feature_cols = [
        c for c in df.columns
        if c not in always_drop
    ]

    return df[feature_cols], y, extras


_SCALERS = {
    "none": "passthrough",
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _make_encoder(encoder_type, seed):
    if encoder_type == "onehot":
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    if encoder_type == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    if encoder_type == "count":
        return CountEncoder(
            handle_unknown=0,
            handle_missing=0
        )

    if encoder_type == "target":
        return TargetEncoder(
            target_type="binary",
            cv=StratifiedKFold(
                5, shuffle=True, random_state=seed
            )
        )

    raise ValueError(f"Unknown encoder: {encoder_type}")


def build_preprocessor(preprocessing_config):
    """Build the preprocessing pipeline from config.yaml."""
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[
        preprocessing_config["scaler"]
    ]
    scaler = (
        scaler_factory()
        if callable(scaler_factory)
        else scaler_factory
    )

    encoder = _make_encoder(
        preprocessing_config["encoder"],
        preprocessing_config.get("random_state")
    )

    numeric_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy=imputation.get(
                    "numeric_strategy", "median"
                )
            )
        ),
        ("scale", scaler),
    ])

    categorical_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy=imputation.get(
                    "categorical_strategy",
                    "most_frequent"
                )
            )
        ),
        ("encode", encoder),
    ])

    indicator_cols = [
        f"{c}_was_missing"
        for c in preprocessing_config.get(
            "mnar_indicator_sources", []
        )
    ]

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            preprocessing_config["numeric_features"]
        ),
        (
            "categorical",
            categorical_pipeline,
            preprocessing_config["categorical_features"]
        ),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_dev_test(X, y, extras, test_size, random_state):
    """Create a development set and a locked final test set."""
    return train_test_split(
        X,
        y,
        extras,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )