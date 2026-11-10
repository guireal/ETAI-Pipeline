import numpy as np

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def adaboost(max_depth=1, **params):
    estimator = DecisionTreeClassifier(max_depth=max_depth)
    return AdaBoostClassifier(estimator=estimator, **params)


def oblique_random_forest(
    n_estimators=100,
    n_components=10,
    density="auto",
    random_state=None,
    n_jobs=None,
    **tree_params
):
    tree_params.setdefault("max_features", "sqrt")

    estimator = Pipeline([
        ("project", SparseRandomProjection(
            n_components=n_components,
            density=density
        )),
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
    "naive_bayes": GaussianNB,
    "xgboost": XGBClassifier
}


def build_model(model_config):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}")

    return _MODELS[model_type](**params)


class VoteSelector(SelectorMixin, BaseEstimator):

    def __init__(self, min_votes=2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        forest = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=self.random_state,
            n_jobs=-1
        )
        forest.fit(X, y)

        booster = GradientBoostingClassifier(
            random_state=self.random_state
        )
        booster.fit(X, y)

        lasso = LogisticRegression(
            l1_ratio=1.0,
            C=0.01,
            solver="liblinear"
        )
        lasso.fit(X, y)

        self.votes_ = (
            (forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
            + (booster.feature_importances_ > booster.feature_importances_.mean())
            + (np.abs(lasso.coef_[0]) > 1e-8)
        )

        self.support_ = self.votes_ >= self.min_votes

        # Keep the feature with the most votes if none reaches the threshold.
        if not np.any(self.support_):
            self.support_[np.argmax(self.votes_)] = True

        return self

    def _get_support_mask(self):
        return self.support_


def build_pipeline(
    preprocessing_config,
    model_config,
    selection_config=None,
    memory=None
):
    steps = [
        ("prep", build_preprocessor(preprocessing_config))
    ]

    if selection_config and selection_config.get("enabled", False):
        steps.append((
            "select",
            VoteSelector(
                min_votes=selection_config.get("min_votes", 2),
                random_state=selection_config.get("random_state")
            )
        ))

    steps.append(("model", build_model(model_config)))

    return Pipeline(steps, memory=memory)