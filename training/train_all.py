"""Downloads the dataset, builds the preprocessor, splits the data, then
trains all 5 classification models. Run from the project root with
`python training/train_all.py`.
"""

import io
import os
import sys
import pathlib
import subprocess
import zipfile
import warnings

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

BASE_DIR = pathlib.Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_CSV = BASE_DIR / "bank-additional-full.csv"

NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
TARGET = "y"
DROP_COLS = ["duration"]  # leaks target


def fetch_dataset() -> pd.DataFrame:
    if RAW_CSV.exists():
        print(f"[data] Reading cached {RAW_CSV.name}")
    else:
        print("[data] Downloading Bank Marketing zip from UCI...")
        try:
            response = requests.get(DATA_URL, timeout=120)
        except Exception:
            warnings.warn("SSL verification failed, retrying without verification.")
            response = requests.get(DATA_URL, timeout=120, verify=False)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # The zip contains a nested zip; extract the inner one
            entry_names = zip_file.namelist()
            print(f"[data] Zip contents: {entry_names}")
            # Look for bank-additional-full.csv directly or inside a nested zip
            if "bank-additional-full.csv" in entry_names:
                RAW_CSV.write_bytes(zip_file.read("bank-additional-full.csv"))
            else:
                # There may be a nested zip like bank-additional.zip
                for entry_name in entry_names:
                    if entry_name.endswith(".zip"):
                        inner_bytes = zip_file.read(entry_name)
                        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
                            inner_entries = inner_zip.namelist()
                            csv_name = next(
                                (n for n in inner_entries if n.endswith("bank-additional-full.csv")),
                                None,
                            )
                            if csv_name:
                                RAW_CSV.write_bytes(inner_zip.read(csv_name))
                                break
        if not RAW_CSV.exists():
            raise FileNotFoundError("Could not extract bank-additional-full.csv from the zip.")
        print(f"[data] Saved to {RAW_CSV}")
    raw_df = pd.read_csv(RAW_CSV, sep=";")
    return raw_df


def make_preprocessor():
    numeric_pipeline = StandardScaler()
    categorical_pipeline = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    feature_pipeline = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return feature_pipeline


def run_pipeline():
    raw_df = fetch_dataset()

    raw_df = raw_df.drop(columns=DROP_COLS)

    label_encoder = LabelEncoder()
    raw_df[TARGET] = label_encoder.fit_transform(raw_df[TARGET])  # no -> 0, yes -> 1

    features = raw_df.drop(columns=[TARGET])
    labels = raw_df[TARGET]

    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    feature_pipeline = make_preprocessor()
    train_x_enc = feature_pipeline.fit_transform(train_x)
    test_x_enc = feature_pipeline.transform(test_x)

    joblib.dump(feature_pipeline, MODEL_DIR / "preprocessor.pkl")
    print("[prep] Preprocessor saved to models/preprocessor.pkl")

    feature_names = NUMERIC_COLS + CATEGORICAL_COLS

    test_df = pd.DataFrame(test_x_enc, columns=feature_names)
    test_df[TARGET] = test_y.values
    test_csv = BASE_DIR / "test_data.csv"
    test_df.to_csv(test_csv, index=False)
    print(f"[data] Test set saved to test_data.csv ({len(test_df)} rows)")

    train_df = pd.DataFrame(train_x_enc, columns=feature_names)
    train_df[TARGET] = train_y.values
    train_csv = BASE_DIR / "train_data.csv"
    train_df.to_csv(train_csv, index=False)
    print(f"[data] Train set saved to train_data.csv ({len(train_df)} rows)\n")

    scripts = [
        "logistic_regression.py",
        "decision_tree.py",
        "knn.py",
        "naive_bayes.py",
        "random_forest.py",
    ]
    for script in scripts:
        script_path = pathlib.Path(__file__).parent / script
        print(f"\nRunning {script}")
        subprocess.run([sys.executable, str(script_path)], check=True)

    print("\n\nAll models trained successfully.")
    print("Saved binaries are in models/")


if __name__ == "__main__":
    run_pipeline()
