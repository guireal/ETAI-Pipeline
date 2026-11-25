"""Cleaning, feature splitting and leak-safe preprocessing."""

import numpy as np
import pandas as pd

from category_encoders import CountEncoder

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import LocalOutlierFactor
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

        numeric = pd.to_numeric(df[column], errors="coerce")

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


def _canonicalize_categories(df, columns_and_maps, placeholder_tokens):
    out = df.copy()

    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue

        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()

        out[col] = lowered.map(mapping).fillna(cleaned)

        out.loc[
            out[col].astype(str).str.strip().isin(placeholder_tokens),
            col
        ] = np.nan

    return out


def clean_dataset(df, diagnostics_config):
    """Clean categories, invalid values and redundant columns."""
    out = df.copy()

    placeholder_tokens = set(
        diagnostics_config.get("placeholder_tokens", [])
    )

    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].replace(list(placeholder_tokens), np.nan),
                errors="coerce"
            )

    for col in diagnostics_config.get("categorical_columns", []):
        if col in out.columns:
            missing = (
                out[col].astype("string").str.strip().isin(placeholder_tokens)
            )
            out.loc[missing, col] = np.nan

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
        c for c in diagnostics_config.get("redundant_columns", [])
        if c in out.columns
    ]

    return out.drop(columns=columns_to_drop)


def drop_duplicate_rows(df, id_column=None):
    """Drop duplicate records before splitting development and test data."""
    out = df.drop_duplicates()

    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")

    return out


def add_missingness_indicators(df, mnar_indicator_sources):
    """Add flags before missing values are imputed."""
    out = df.copy()

    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)

    return out


def split_features_target(df, data_config, mnar_indicator_sources):
    """Separate features, target and columns reserved for fairness auditing."""
    target = data_config["target"]
    sensitive_attr = data_config.get("sensitive_attr")
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)

    y = df[target] if target in df.columns else None

    extras_cols = [
        c for c in [sensitive_attr, "score_text"]
        if c and c in df.columns
    ]

    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target}

    if sensitive_attr:
        always_drop.add(sensitive_attr)

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


_NUMERIC_IMPUTERS = {
    "median": lambda cfg, seed: SimpleImputer(
        strategy="median"
    ),

    "mean": lambda cfg, seed: SimpleImputer(
        strategy="mean"
    ),

    "knn": lambda cfg, seed: KNNImputer(
        n_neighbors=cfg.get("n_neighbors", 5)
    ),

    "iterative": lambda cfg, seed: IterativeImputer(
        max_iter=cfg.get("max_iter", 10),
        random_state=seed
    ),

    "random_forest": lambda cfg, seed: IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=cfg.get("n_estimators", 50),
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1
        ),
        max_iter=cfg.get("max_iter", 10),
        random_state=seed
    ),
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
            cv=StratifiedKFold(5, shuffle=True, random_state=seed)
        )

    raise ValueError(f"Unknown encoder: {encoder_type}")


def build_preprocessor(preprocessing_config):
    """Build the preprocessing pipeline from the configuration."""
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[preprocessing_config["scaler"]]

    scaler = (
        scaler_factory()
        if callable(scaler_factory)
        else scaler_factory
    )

    encoder = _make_encoder(
        preprocessing_config["encoder"],
        preprocessing_config.get("random_state")
    )

    numeric_strategy = imputation.get(
        "numeric_strategy", "median"
    )

    if numeric_strategy not in _NUMERIC_IMPUTERS:
        raise ValueError(
            f"Unknown numeric_strategy: {numeric_strategy}. "
            f"Options: {list(_NUMERIC_IMPUTERS)}"
        )

    numeric_imputer = _NUMERIC_IMPUTERS[numeric_strategy](
        imputation,
        preprocessing_config.get("random_state")
    )

    numeric_pipeline = Pipeline([
        ("scale", scaler),
        ("impute", numeric_imputer),
    ])

    categorical_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy=imputation.get(
                    "categorical_strategy", "most_frequent"
                )
            )
        ),
        ("encode", encoder),
    ])

    indicator_cols = [
        f"{c}_was_missing"
        for c in preprocessing_config.get("mnar_indicator_sources", [])
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


class AnomalyScore(TransformerMixin, BaseEstimator):
    """
    Adds an anomaly_score feature to the data.

    Available methods:
    - isolation_forest
    - lof

    Higher scores indicate more unusual observations.
    """

    def __init__(self, method="isolation_forest", n_neighbors=20, random_state=None):
        self.method = method
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.method == "isolation_forest":
            self.detector_ = IsolationForest(
                random_state=self.random_state
            )

        elif self.method == "lof":
            self.detector_ = LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                novelty=True
            )

        else:
            raise ValueError(
                f"Unknown anomaly method: {self.method}. "
                "Options: isolation_forest, lof"
            )

        self.detector_.fit(X)
        self.n_features_in_ = np.asarray(X).shape[1]

        return self

    def fit_transform(self, X, y=None):
        self.fit(X)

        if self.method == "lof":
            scores = -self.detector_.negative_outlier_factor_
        else:
            scores = -self.detector_.score_samples(X)

        return np.column_stack([X, scores])

    def transform(self, X):
        scores = -self.detector_.score_samples(X)

        return np.column_stack([X, scores])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [
                f"x{i}" for i in range(self.n_features_in_)
            ]

        return np.append(
            np.asarray(input_features, dtype=object),
            "anomaly_score"
        )


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