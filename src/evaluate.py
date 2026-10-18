"""Cross-validation, classification and fairness evaluation."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate


def cross_validate_pipeline(
    pipeline, X, y, cv, scoring="accuracy", n_jobs=1
):
    """Evaluate a pipeline and return fold scores and OOF predictions."""
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs
    )

    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"]
    })

    fold_scores["gap"] = (
        fold_scores["train"] - fold_scores["validation"]
    )

    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)

    for model, val_idx in zip(
        scores["estimator"],
        scores["indices"]["test"]
    ):
        y_oof[val_idx] = model.predict(X.iloc[val_idx])

    return fold_scores, y_oof


def cv_report(fold_scores, scoring="accuracy"):
    """Display the cross-validation results."""
    lines = [
        f"Cross-validation ({len(fold_scores)} folds, metric: {scoring})",
        "",
        fold_scores.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}"
        ),
        ""
    ]

    for column in ["train", "validation", "gap"]:
        lines.append(
            f"{column.capitalize()} mean = "
            f"{fold_scores[column].mean():.3f}; "
            f"std = {fold_scores[column].std(ddof=1):.3f}"
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


def oof_classification_report(y_true, y_pred):
    """Classification report using out-of-fold predictions."""
    text = (
        "Classification report (out-of-fold predictions, "
        "development set):\n"
        + classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    print(text)
    return text


def fairness_report(
    y_true, y_pred, extras, sensitive_attr="race"
):
    """Compare false-positive rates by group."""
    df = extras.copy()

    df[sensitive_attr] = (
        df[sensitive_attr].astype(object).fillna("unknown")
    )
    df["y_true"] = np.asarray(y_true)
    df["y_pred_model"] = np.asarray(y_pred)
    df["y_pred_compas"] = (
        df["score_text"] != "Low"
    ).astype(int)

    lines = [
        f"False positive rate by {sensitive_attr}",
        "(development set, out-of-fold predictions)",
        ""
    ]

    for label, column in [
        ("Our model", "y_pred_model"),
        ("COMPAS score", "y_pred_compas")
    ]:
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