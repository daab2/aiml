# Program 7: Implementation of KNN (manual pages 38-40).
# Aim: Predict an Iris flower's species using K=3 nearest neighbours.
# Theory: KNN is a supervised, non-parametric, lazy learning algorithm.
# It stores labelled examples and predicts by a majority vote among neighbours.
# Description of the Data Set:
# Name of the Data Set: Iris (iris.csv beside this file).
# Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/data/iris.csv
# CSV: header added, labels named; original measurements retained (data only).
# Requires NumPy: python3 -m pip install numpy
# Run: python3 prog7.py
# No. of Features: 4 (sepal length/width and petal length/width, in cm).
# No. of Instances: 150 (50 each of setosa, versicolor and virginica).
# No. of Numerical Data: 4 numerical input columns.
# No. of Categorical Data: 0 input columns; 1 categorical target column.
# Name of the Target Variable: species.
# Algorithm: Store training examples; compute Euclidean distances; sort them;
# select the first K neighbours; count votes; return the class with most votes.
# Ties in votes are resolved alphabetically; equal distances keep training order.
# Use a seeded 80/20 split within each species and training-only standardization.
# Output: Actual and predicted test classes and test accuracy.
# Pros: Simple, supports nonlinear boundaries and requires no fitted parameters.
# Cons: Stores all training data; prediction is slow and sensitive to feature scale.

import csv
from pathlib import Path
import numpy as np


def knn(point, X_train, y_train, k=3):
    distances = []
    for sample, label in zip(X_train, y_train):
        distance = np.sqrt(np.sum((point - sample) ** 2))
        distances.append((distance, label))
    distances.sort(key=lambda item: item[0])
    votes = {}
    for distance, label in distances[:k]:
        votes[label] = votes.get(label, 0) + 1
    return min(votes, key=lambda label: (-votes[label], label))


if __name__ == "__main__":
    with (Path(__file__).parent / "iris.csv").open(newline="") as file:
        rows = list(csv.reader(file))[1:]
    X = np.array([row[:4] for row in rows], dtype=float)
    y = np.array([row[4] for row in rows])
    rng = np.random.default_rng(42)
    train, test = [], []
    for label in sorted(set(y)):
        indices = rng.permutation(np.where(y == label)[0])
        split = int(0.8 * len(indices))
        train.extend(indices[:split])
        test.extend(indices[split:])
    mean, scale = X[train].mean(axis=0), X[train].std(axis=0)
    scale[scale == 0] = 1
    X_train, X_test = (X[train] - mean) / scale, (X[test] - mean) / scale
    predicted = np.array([knn(point, X_train, y[train]) for point in X_test])
    print(f"K = 3 | Training: {len(train)} | Test: {len(test)}")
    print(f"{'Actual':15} Predicted")
    for actual, prediction in zip(y[test], predicted):
        print(f"{actual:15} {prediction}")
    print(f"Accuracy: {np.mean(predicted == y[test]) * 100:.2f}%")
