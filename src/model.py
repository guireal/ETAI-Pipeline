"""Model construction."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier


def adaboost(max_depth=1, **params):
    """AdaBoost using decision trees."""
    estimator = DecisionTreeClassifier(max_depth=max_depth)

    return AdaBoostClassifier(
        estimator=estimator,
        **params
    )


def oblique_random_forest(
    n_estimators=100,
    n_components=10,
    density="auto",
    random_state=None,
    n_jobs=None,
    **tree_params
):
    """Bagged decision trees using sparse random projections."""
    tree_params.setdefault("max_features", "sqrt")

    estimator = Pipeline([
        (
            "project",
            SparseRandomProjection(
                n_components=n_components,
                density=density,
                random_state=random_state
            )
        ),
        ("tree", DecisionTreeClassifier(**tree_params))
    ])

    return BaggingClassifier(
        estimator=estimator,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB
}


def build_model(model_config):
    """Build a model using config.yaml."""
    model_type = model_config["type"]
    params = model_config.get("params", {}).copy()

    if model_type not in _MODELS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Options: {list(_MODELS)}"
        )

    return _MODELS[model_type](**params)


def build_pipeline(preprocessing_config, model_config):
    """Build a complete preprocessing and model pipeline."""
    from src.preprocessing import build_preprocessor

    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config))
    ])