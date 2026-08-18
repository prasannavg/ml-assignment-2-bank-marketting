# Bank Marketing - ML Classification Assignment

## Problem Statement

Predict whether a bank client will subscribe to a term deposit product
based on direct marketing campaign data (phone calls).
This is a **binary classification** problem: output `yes` (1) or `no` (0).

---

## Dataset Description

| Property | Value |
|---|---|
| Source | [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) |
| File | `bank-additional-full.csv` |
| Instances | 41,188 |
| Features | 20 (19 used after dropping `duration` which leaks the target) |
| Target | `y`, has the client subscribed to a term deposit? (yes/no) |
| Class balance | ~88.7% No, ~11.3% Yes (imbalanced) |

**Feature groups:**

- **Numeric (9):** age, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
- **Categorical (10):** job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome

---

## GitHub Repository Link

> **[https://github.com/prasannavg/ml-assignment-2-bank-marketting]https://github.com/prasannavg/ml-assignment-2-bank-marketting)**
>

---

## Models Used

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9007 | 0.7954 | 0.7022 | 0.2058 | 0.3183 | 0.3446 |
| Decision Tree | 0.8985 | 0.7609 | 0.6065 | 0.2823 | 0.3853 | 0.3674 |
| kNN | 0.8978 | 0.7438 | 0.6064 | 0.2640 | 0.3679 | 0.3547 |
| Naive Bayes | 0.8398 | 0.7731 | 0.3548 | 0.5162 | 0.4205 | 0.3391 |
| Random Forest (Ensemble) | 0.9019 | 0.8089 | 0.6546 | 0.2737 | 0.3860 | 0.3811 |

*Evaluated on 20% stratified test split (8,238 rows). Dataset is imbalanced (~88.7% No / 11.3% Yes), so AUC and MCC are better discriminators than raw accuracy.*

---

## Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | High precision (70.2%) but very low recall (20.6%). The model is overly conservative on the imbalanced dataset: it identifies non-subscribers well but misses 80% of actual subscribers. Best when false positives are costly. |
| Decision Tree | Moderate and balanced performance (Accuracy 89.85%, AUC 0.76). Better recall (28.2%) than LR at the cost of some precision. Fully interpretable, each prediction path can be traced. |
| kNN | Similar to Decision Tree but marginally lower AUC (0.74). Benefits from StandardScaler preprocessing. Slower at prediction time for large datasets since it computes distances to all training points. |
| Naive Bayes | Lowest accuracy (83.98%) but highest recall (51.6%), catching more than twice as many actual subscribers as LR. The Gaussian feature-independence assumption does not perfectly fit all correlated features, trading accuracy for sensitivity. |
| Random Forest (Ensemble) | Best overall: highest accuracy (90.19%), best AUC (80.89%), best MCC (0.3811). Ensemble of 100 trees reduces overfitting and generalises well. The recommended model for deployment. |
| **Overall Winner** | **Random Forest**, leads on Accuracy, AUC, and MCC. For recall-sensitive use cases (e.g. maximising outreach), Naive Bayes is the runner-up. |

---

## Live Streamlit App

> **[https://ml-assignment-2-2025ac05356.streamlit.app/](https://ml-assignment-2-2025ac05356.streamlit.app/)**
>

---

## Project Structure

```
assignment-2/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── test_data.csv          # Test split (20%) used in the app
├── train_data.csv         # Train split (80%), generated locally
├── models/                # Saved model files (*.pkl), loaded by the app
│   ├── preprocessor.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── random_forest.pkl
└── training/               # Training source code (*.ipynb)
    └── train_all.ipynb    # Loads data, preprocesses, trains all 5 models, saves models/*.pkl
```

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (downloads dataset, saves test_data.csv + models/*.pkl)
jupyter nbconvert --to notebook --execute --inplace training/train_all.ipynb
# (or open training/train_all.ipynb in Jupyter and Run All)

# 3. Launch the Streamlit app
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (include `models/*.pkl` and `test_data.csv`)
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub, then New App, select the repo, branch `main`, and `app.py`
4. Click **Deploy**
