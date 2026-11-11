
"""Cross-validation, classification and fairness evaluation."""

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    get_scorer
)
from sklearn.model_selection import cross_validate


def build_scorers(metrics):
    """Build the scoring functions specified in the configuration."""
    scorers = {}

    for name in metrics:
        if name.startswith("fbeta_"):
            beta = float(name.split("_")[1])
            scorers[name] = make_scorer(
                fbeta_score,
                beta=beta,
                zero_division=0
            )
        else:
            scorers[name] = get_scorer(name)

    return scorers


def cross_validate_pipeline(
    pipeline, X, y, cv, scorers, selection_metric, n_jobs=1
):
    """Evaluate a pipeline and return fold scores and OOF probabilities."""
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scorers,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs
    )

    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["estimator"]) + 1),
        "train": scores[f"train_{selection_metric}"],
        "validation": scores[f"test_{selection_metric}"]
    })

    fold_scores["gap"] = (
        fold_scores["train"] - fold_scores["validation"]
    )

    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X), dtype=float)

    for model, val_idx in zip(
        scores["estimator"],
        scores["indices"]["test"]
    ):
        oof_proba[val_idx] = model.predict_proba(
            X.iloc[val_idx]
        )[:, 1]

    return fold_scores, oof_proba


def cv_report(fold_scores, metrics, selection_metric, title):
    """Display the cross-validation results."""
    lines = [
        f"{title} ({len(fold_scores)} folds)",
        f"Selection metric: {selection_metric}",
        "",
        fold_scores.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}"
        ),
        ""
    ]

    for metric in metrics:
        lines.append(
            f"{metric}: mean = {fold_scores[metric].mean():.3f}; "
            f"std = {fold_scores[metric].std(ddof=1):.3f}"
        )

    text = "\n".join(lines)
    print(text)
    return text


def corrected_paired_test(scores_a, scores_b, test_train_ratio):
    """Corrected paired t-test for scores from the same CV folds."""
    differences = (
        np.asarray(scores_a, dtype=float)
        - np.asarray(scores_b, dtype=float)
    )

    if len(differences) < 2:
        raise ValueError(
            "At least two paired fold scores are required."
        )

    standard_error = np.sqrt(
        (1 / len(differences) + test_train_ratio)
        * differences.var(ddof=1)
    )

    if standard_error > 0:
        t_value = float(differences.mean() / standard_error)
    else:
        t_value = 0.0

    p_value = float(
        2 * stats.t.sf(
            abs(t_value),
            df=len(differences) - 1
        )
    )

    return {
        "mean_diff": float(differences.mean()),
        "wins": (
            f"{int((differences > 0).sum())}/"
            f"{len(differences)}"
        ),
        "t": t_value,
        "p_value": p_value
    }


def oof_classification_report(y_true, oof_proba, threshold=0.5):
    """Classification report using out-of-fold probabilities."""
    y_pred = (
        np.asarray(oof_proba) >= threshold
    ).astype(int)

    matrix = confusion_matrix(y_true, y_pred)

    text = (
        "Confusion matrix (out-of-fold predictions):\n"
        f"{matrix}\n\n"
        "Classification report (development set):\n"
        + classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    print(text)
    return text


def fairness_report(
    y_true,
    oof_proba,
    extras,
    sensitive_attr=None,
    threshold=0.5
):
    """Calculate false-positive rates for groups in the sensitive attribute."""
    if extras is None or sensitive_attr is None:
        return "Fairness report unavailable: no sensitive attribute configured."

    if sensitive_attr not in extras.columns:
        return (
            f"Fairness report unavailable: "
            f"'{sensitive_attr}' was not found."
        )

    df = extras.copy()

    df[sensitive_attr] = (
        df[sensitive_attr].astype(object).fillna("unknown")
    )

    df["y_true"] = np.asarray(y_true)
    df["y_pred_model"] = (
        np.asarray(oof_proba) >= threshold
    ).astype(int)

    # COMPAS has its own risk category for a direct comparison.
    has_compas_score = "score_text" in df.columns

    if has_compas_score:
        df["y_pred_compas"] = (
            df["score_text"].astype(str).str.strip().str.lower() != "low"
        ).astype(int)

    lines = [
        f"False positive rate by {sensitive_attr}",
        "(development set, out-of-fold predictions)",
        ""
    ]

    columns = [("Our model", "y_pred_model")]

    if has_compas_score:
        columns.append(("COMPAS score", "y_pred_compas"))

    for label, column in columns:
        lines.append(f"  {label}:")

        for group, group_df in df.groupby(
            sensitive_attr, dropna=False
        ):
            negatives = group_df[group_df["y_true"] == 0]

            if len(negatives) == 0:
                continue

            fpr = (negatives[column] == 1).mean()

            lines.append(
                f"    {str(group):<20s} FPR = {fpr:.2f} "
                f"(n={len(negatives)})"
            )

        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text
