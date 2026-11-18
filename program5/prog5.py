# Program 5: Implement binary logistic regression (manual pages 31-33).
# Theory: sigmoid(w.x+b) estimates P(y=1); threshold 0.5 predicts the class.
# Algorithm (p32): initialize small random weights/bias; for each iteration,
# visit each training example, calculate sigmoid/log loss, then update w and b.
# Keep the manual's per-example gradients dw=x*(a-y)/m and db=(a-y)/m.
# Name of the Data Set: Iris, setosa/versicolor subset (iris.csv beside this file).
# No. of Features: 4 (sepal length/width, petal length/width, all in cm).
# No. of Instances: 100 used (50 per class); full CSV has 150.
# No. of Numerical Data: 4 input columns (400 numeric feature values used).
# No. of Categorical Data: 0 input columns; 1 categorical target column.
# Name of the Target Variable: species (0=setosa, 1=versicolor).
# Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/data/iris.csv
# CSV: header added, labels named; original measurements retained (data only).
# Requires NumPy: python3 -m pip install numpy
# Run: python3 prog5.py
# Split: seeded 80 training / 20 testing (80/20 per class); standardize using
# training means/stds only. NumPy supplies arithmetic; the model is built here.
# Input: local CSV. Output: actual/predicted test species, accuracy, log loss.
# Advantage: simple probability estimates. Limitations: linear decision
# boundary; correlated features can affect interpretation of the weights.

import csv
from pathlib import Path
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def train(x, y, rate=0.1, iterations=2000):
    rng = np.random.default_rng(0)
    w, b = rng.normal(0, 0.01, x.shape[1]), rng.normal(0, 0.01)
    m = len(y)
    for _ in range(iterations):
        for sample, target in zip(x, y):
            z = sample @ w + b
            a = sigmoid(z)
            cost = -(target*np.log(a + 1e-15) + (1-target)*np.log(1-a + 1e-15))
            dw = sample * (a - target) / m
            db = (a - target) / m
            w -= rate * dw
            b -= rate * db
    return w, b


if __name__ == "__main__":
    with (Path(__file__).parent / "iris.csv").open() as file:
        rows = [r for r in csv.DictReader(file) if r["species"] != "virginica"]
    names = np.array(["setosa", "versicolor"])
    x = np.array([[float(r[k]) for k in list(r)[:4]] for r in rows])
    y = np.array([int(r["species"] == "versicolor") for r in rows])
    rng, train_ids, test_ids = np.random.default_rng(42), [], []
    for c in (0, 1):
        ids = rng.permutation(np.flatnonzero(y == c))
        train_ids.extend(ids[:40]); test_ids.extend(ids[40:])
    mean, scale = x[train_ids].mean(axis=0), x[train_ids].std(axis=0)
    x = (x - mean) / np.where(scale == 0, 1, scale)
    w, b = train(x[train_ids], y[train_ids])
    p = sigmoid(x[train_ids] @ w + b)
    loss = -np.mean(y[train_ids]*np.log(p + 1e-15) + (1-y[train_ids])*np.log(1-p + 1e-15))
    predicted = (sigmoid(x[test_ids] @ w + b) >= 0.5).astype(int)
    print("Actual       Predicted")
    for actual, guess in zip(y[test_ids], predicted):
        print(f"{names[actual]:12} {names[guess]}")
    print(f"Accuracy: {np.mean(predicted == y[test_ids]):.2%}")
    print(f"Training log loss: {loss:.4f}")
