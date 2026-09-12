"""
models.py
=========
Train and evaluate four model configurations:

    1. Raw features      + Logistic Regression
    2. Raw features      + Random Forest
    3. Engineered features + Logistic Regression
    4. Engineered features + Random Forest

Each model reports accuracy, precision, recall, F1, and its confusion matrix
on the TEST set.  Training-set accuracy is also printed to flag overfitting.

Random seed (RANDOM_SEED = 42) is set on every model for reproducibility.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RANDOM_SEED = 42


def _make_logistic_regression() -> LogisticRegression:
    """Logistic Regression with L2 regularisation and a generous iteration cap."""
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )


def _make_random_forest() -> RandomForestClassifier:
    """
    Random Forest with depth/leaf constraints to limit overfitting.

    max_depth=5 and min_samples_leaf=20 prevent the trees from memorising
    the training set, which is critical for noisy financial data.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


_MODEL_FACTORIES = {
    "LogisticRegression": _make_logistic_regression,
    "RandomForest":       _make_random_forest,
}


def _train_and_evaluate(
    X_train: np.ndarray,
    y_train: "pd.Series",
    X_test: np.ndarray,
    y_test: "pd.Series",
    feature_set: str,
    model_name: str,
    model,
) -> dict:
    """
    Fit `model` on training data and evaluate on test data.

    Prints training accuracy alongside test metrics so overfitting is visible.

    Returns
    -------
    dict with label, metrics, confusion matrix, predictions, and model object.
    """
    model.fit(X_train, y_train.values)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_acc = accuracy_score(y_train.values, y_pred_train)

    acc  = accuracy_score(y_test.values,  y_pred_test)
    prec = precision_score(y_test.values, y_pred_test, zero_division=0)
    rec  = recall_score(y_test.values,    y_pred_test, zero_division=0)
    f1   = f1_score(y_test.values,        y_pred_test, zero_division=0)
    cm   = confusion_matrix(y_test.values, y_pred_test)

    label = f"{feature_set} + {model_name}"
    print(
        f"  {label:<42} "
        f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  "
        f"[TrainAcc={train_acc:.4f}]"
    )
    print(f"    Confusion matrix (rows=actual, cols=pred):\n{cm}\n")

    return {
        "label":          label,
        "feature_set":    feature_set,
        "model_name":     model_name,
        "Accuracy":       acc,
        "Precision":      prec,
        "Recall":         rec,
        "F1":             f1,
        "train_accuracy": train_acc,
        "confusion_matrix": cm,
        "y_pred":         y_pred_test,
        "model":          model,
    }


def run_all_models(splits_raw: dict, splits_eng: dict) -> list:
    """
    Train and evaluate all four model × feature-set combinations.

    Order (matches the required results table):
        Raw         + LogisticRegression
        Raw         + RandomForest
        Engineered  + LogisticRegression
        Engineered  + RandomForest

    Parameters
    ----------
    splits_raw : output of split_and_scale() for raw features
    splits_eng : output of split_and_scale() for engineered features

    Returns
    -------
    list of result dicts (4 items)
    """
    results = []

    print("\n--- Model Evaluation (TrainAcc shown to check for overfitting) ---\n")

    split_map = [
        ("Raw",        splits_raw),
        ("Engineered", splits_eng),
    ]

    for feature_set, splits in split_map:
        for model_name, factory in _MODEL_FACTORIES.items():
            result = _train_and_evaluate(
                X_train=splits["X_train"],
                y_train=splits["y_train"],
                X_test=splits["X_test"],
                y_test=splits["y_test"],
                feature_set=feature_set,
                model_name=model_name,
                model=factory(),   # fresh instance each time
            )
            results.append(result)

    return results
