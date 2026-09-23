import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MNAR_INDICATOR_SOURCES = ["priors_count", "c_charge_degree"]
NUMERIC_FEATURES = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
CATEGORICAL_FEATURES = ["sex", "age_cat", "c_charge_degree"]

CANONICAL_MAPS = {
    "sex": {"male": "Male", "female": "Female"},
    "race": {
        "african-american": "African-American",
        "african american": "African-American",
        "caucasian": "Caucasian",
        "hispanic": "Hispanic",
        "other": "Other",
        "asian": "Asian",
        "native american": "Native American",
    },
    "c_charge_degree": {"f": "F", "felony": "F", "m": "M", "misdemeanor": "M"},
    "score_text": {"low": "Low", "medium": "Medium", "high": "High"},
}

PLACEHOLDER_TOKENS = {"-", "?", "n/a", "N/A", "na", "NA", ""}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes categories, handles domain rule NaNs, and drops duplicates/redundant columns."""
    out = df.copy()

    # 1. Clean strings & placeholders
    for col in ["priors_count", "prior_offenses"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(PLACEHOLDER_TOKENS), None), errors="coerce")

    for col, mapping in CANONICAL_MAPS.items():
        if col in out.columns:
            cleaned = out[col].astype(str).str.strip().str.lower()
            out[col] = cleaned.map(mapping).fillna(out[col])
            out.loc[out[col].astype(str).str.strip().isin(PLACEHOLDER_TOKENS), col] = None

    # 2. Domain rules: replace impossible values with NaN
    if "age" in out.columns:
        out.loc[~pd.to_numeric(out["age"], errors="coerce").between(18, 100), "age"] = None
    if "decile_score" in out.columns:
        out.loc[~pd.to_numeric(out["decile_score"], errors="coerce").between(1, 10), "decile_score"] = None
    if "juv_fel_count" in out.columns:
        out.loc[pd.to_numeric(out["juv_fel_count"], errors="coerce") < 0, "juv_fel_count"] = None
    if "priors_count" in out.columns:
        out.loc[~pd.to_numeric(out["priors_count"], errors="coerce").between(0, 60), "priors_count"] = None

    # 3. Deduplicate
    out = out.drop_duplicates()
    if "id" in out.columns:
        out = out.drop_duplicates(subset="id", keep="first")

    # 4. Drop multicollinear columns
    cols_to_drop = [c for c in ["prior_offenses", "age_in_months", "juvenile_total"] if c in out.columns]
    out = out.drop(columns=cols_to_drop)

    return out


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn preprocessor with imputation and one-hot encoding."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    numeric_indicators = [f"{c}_was_missing" for c in MNAR_INDICATOR_SOURCES]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("indicators", "passthrough", numeric_indicators),
    ])


def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    # Add MNAR indicators
    for col in MNAR_INDICATOR_SOURCES:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)

    y = df[target]
    extras = df[[sensitive_attr, "score_text"]].copy()

    drop_always = [target, sensitive_attr] + [c for c in drop_columns if c in df.columns]
    X = df.drop(columns=drop_always)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test, extras_test