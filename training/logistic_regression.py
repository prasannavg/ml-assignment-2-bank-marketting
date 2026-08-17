"""Logistic Regression classifier on the Bank Marketing dataset."""

import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

BASE_DIR = pathlib.Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_TAG = "logistic_regression"


def read_datasets():
    feature_pipeline = joblib.load(MODEL_DIR / "preprocessor.pkl")
    train_df = pd.read_csv(BASE_DIR / "train_data.csv")
    test_df = pd.read_csv(BASE_DIR / "test_data.csv")
    train_x = train_df.drop(columns=["y"]).values
    train_y = train_df["y"].values
    test_x = test_df.drop(columns=["y"]).values
    test_y = test_df["y"].values
    return train_x, train_y, test_x, test_y


def score_predictions(clf, test_x, test_y):
    predicted_y = clf.predict(test_x)
    proba = clf.predict_proba(test_x)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(test_y, predicted_y), 4),
        "AUC":       round(roc_auc_score(test_y, proba), 4),
        "Precision": round(precision_score(test_y, predicted_y, zero_division=0), 4),
        "Recall":    round(recall_score(test_y, predicted_y, zero_division=0), 4),
        "F1":        round(f1_score(test_y, predicted_y, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(test_y, predicted_y), 4),
    }


def run_training():
    train_x, train_y, test_x, test_y = read_datasets()

    clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(train_x, train_y)

    scores = score_predictions(clf, test_x, test_y)
    joblib.dump(clf, MODEL_DIR / f"{MODEL_TAG}.pkl")

    print(f"[{MODEL_TAG}] Metrics:")
    for metric_name, metric_value in scores.items():
        print(f"  {metric_name}: {metric_value}")
    print(f"  Model saved to models/{MODEL_TAG}.pkl")
    return scores


if __name__ == "__main__":
    run_training()
