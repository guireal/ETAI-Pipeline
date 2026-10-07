"""Model construction."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {}).copy()

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS.keys())}")

    if model_type == "dummy" and "strategy" not in params:
        params["strategy"] = "most_frequent"

    return _MODELS[model_type](**params)


def build_pipeline(preprocessor, model) -> Pipeline:
    """Builds an integrated preprocessing and modeling pipeline."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
