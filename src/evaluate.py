"""Evaluation -- stratified k-fold cross-validation, out-of-fold metrics, and fairness audit."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report


def cross_validate_pipeline(pipeline, X_dev, y_dev, n_splits: int = 5, random_state: int = 42):
    """Performs stratified cross-validation on the development set."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_accs, val_accs, gaps = [], [], []
    oof_preds = pd.Series(index=X_dev.index, dtype=int)

    lines = [
        "Cross-Validation Results (Development Set)",
        f"{'Fold':<6}{'Train Acc':<12}{'Val Acc':<12}{'Gap (Train - Val)':<18}",
        "-" * 48,
    ]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
        fold_pipeline = clone(pipeline)

        X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_va, y_va = X_dev.iloc[val_idx], y_dev.iloc[val_idx]

        fold_pipeline.fit(X_tr, y_tr)

        train_pred = fold_pipeline.predict(X_tr)
        val_pred = fold_pipeline.predict(X_va)

        oof_preds.iloc[val_idx] = val_pred

        tr_acc = accuracy_score(y_tr, train_pred)
        va_acc = accuracy_score(y_va, val_pred)
        gap = tr_acc - va_acc

        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        gaps.append(gap)

        lines.append(f"{fold:<6}{tr_acc:<12.3f}{va_acc:<12.3f}{gap:<+18.3f}")

    lines.append("-" * 48)
    lines.append(
        f"Mean: {np.mean(train_accs):.3f} +/- {np.std(train_accs):.3f} (train)   "
        f"{np.mean(val_accs):.3f} +/- {np.std(val_accs):.3f} (val)   "
        f"Gap: {np.mean(gaps):+.3f}"
    )
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)
    return oof_preds, report_text


def evaluate_oof(y_dev, oof_preds) -> str:
    """Generates classification metrics using out-of-fold predictions."""
    lines = [
        "Classification report (test set):",
        classification_report(y_dev, oof_preds),
    ]
    text = "\n".join(lines)
    print(text)
    return text


def fairness_report_oof(y_dev, oof_preds, extras_dev: pd.DataFrame, sensitive_attr: str = "race") -> str:
    """Computes false positive rates by group using out-of-fold predictions."""
    df = extras_dev.copy()
    df["y_true"] = y_dev.values
    df["y_pred_model"] = oof_preds.values
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        "False positive rate by race",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {str(group):<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text