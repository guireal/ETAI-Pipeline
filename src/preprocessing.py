"""Preprocessing and dataset splitting."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, TargetEncoder


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """Standardizes categories, handles domain rule NaNs, and drops redundant columns."""
    out = df.copy()
    tokens = set(diagnostics_config.get("placeholder_tokens", []))
    numeric_text_cols = diagnostics_config.get("numeric_text_columns", [])
    validity_rules = diagnostics_config.get("validity_rules", {})
    canonical_maps = diagnostics_config.get("canonical_categories", {})
    redundant_cols = diagnostics_config.get("redundant_columns", [])

    for col in numeric_text_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(tokens), np.nan), errors="coerce")

    for col, mapping in canonical_maps.items():
        if col in out.columns:
            cleaned = out[col].astype(str).str.strip().str.lower()
            out[col] = cleaned.map(mapping).fillna(out[col])
            out.loc[out[col].astype(str).str.strip().isin(tokens), col] = np.nan

    for col, rule in validity_rules.items():
        if col in out.columns:
            col_series = pd.to_numeric(out[col], errors="coerce")
            if "min" in rule and "max" in rule:
                out.loc[~col_series.between(rule["min"], rule["max"]), col] = np.nan
            elif "min" in rule:
                out.loc[col_series < rule["min"], col] = np.nan
            elif "max" in rule:
                out.loc[col_series > rule["max"], col] = np.nan

    cols_to_drop = [c for c in redundant_cols if c in out.columns]
    out = out.drop(columns=cols_to_drop)

    return out


def drop_duplicate_rows(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Removes duplicate rows prior to splitting."""
    out = df.drop_duplicates()
    if id_col in out.columns:
        out = out.drop_duplicates(subset=id_col, keep="first")
    return out


def build_preprocessor(prep_config: dict) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer from config settings."""
    encoder_type = prep_config.get("encoder", "target")
    scaler_type = prep_config.get("scaler", "robust")
    num_strat = prep_config.get("imputation", {}).get("numeric_strategy", "median")
    cat_strat = prep_config.get("imputation", {}).get("categorical_strategy", "most_frequent")

    numeric_features = prep_config.get("numeric_features", [])
    categorical_features = prep_config.get("categorical_features", [])
    mnar_sources = prep_config.get("mnar_indicator_sources", [])

    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = "passthrough"

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=num_strat)),
        ("scale", scaler),
    ])

    if encoder_type == "target":
        encoder = TargetEncoder(cv=5, smooth="auto")
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=cat_strat)),
        ("encode", encoder),
    ])

    numeric_indicators = [f"{c}_was_missing" for c in mnar_sources]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", numeric_indicators),
    ])


def split_dev_test(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    mnar_sources: list,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Partitions the dataset into development and final test sets."""
    df_proc = df.copy()

    for col in mnar_sources:
        if col in df_proc.columns:
            df_proc[f"{col}_was_missing"] = df_proc[col].isna().astype(int)

    y = df_proc[target]
    extras = df_proc[[sensitive_attr, "score_text"]].copy()

    drop_always = [target, sensitive_attr] + [c for c in drop_columns if c in df_proc.columns]
    X = df_proc.drop(columns=drop_always)

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_dev, y_dev, extras_dev, X_test, y_test, extras_test