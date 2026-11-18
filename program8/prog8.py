# Program 8: Implementation of K-Means clustering (manual pages 41-43).
# Aim: Group Iris flowers into K=3 clusters and plot the clusters.
# Theory: K-Means is unsupervised; it reduces within-cluster squared distances
# by alternating nearest-centroid assignments and mean-centroid updates.
# Description of the Data Set:
# Name of the Data Set: Iris (iris.csv beside this file).
# Source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/data/iris.csv
# CSV: header added, labels named; original measurements retained (data only).
# Requires NumPy and Matplotlib: python3 -m pip install numpy matplotlib
# Run: python3 prog8.py
# No. of Features: 4 (sepal length/width and petal length/width, in cm).
# No. of Instances: 150.
# No. of Numerical Data: 4 numerical input columns.
# No. of Categorical Data: 0 input columns; species is ignored reference data.
# Name of the Target Variable: None (unsupervised clustering).
# Algorithm: Choose K random data points as centroids; assign each point to
# its nearest centroid; update each centroid to its cluster mean; repeat until
# centroid changes are below 0.000001 or 100 iterations. Keep empty centroids.
# All four features are standardized; the plot shows a projection onto the
# first two features in original cm. Cluster numbers do not name species.
# Output: Cluster sizes, centroids and prog8_clusters.png beside this file.
# Pros: Simple, efficient and easy to interpret.
# Cons: Requires K; sensitive to initialization, scale, outliers and local optima.

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def kmeans(X, k=3, max_iter=100):
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    for iteration in range(1, max_iter + 1):
        labels = np.array([np.argmin(np.sum((centroids - point) ** 2, axis=1))
                           for point in X])
        updated = centroids.copy()
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        change = np.max(np.abs(updated - centroids))
        centroids = updated
        if change < 1e-6:
            break
    labels = np.array([np.argmin(np.sum((centroids - point) ** 2, axis=1))
                       for point in X])
    return centroids, labels, iteration


if __name__ == "__main__":
    base = Path(__file__).parent
    with (base / "iris.csv").open(newline="") as file:
        rows = list(csv.reader(file))[1:]
    X = np.array([row[:4] for row in rows], dtype=float)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1
    centroids, labels, iterations = kmeans((X - mean) / scale)
    centroids = centroids * scale + mean
    print(f"K = 3 | Iterations: {iterations}")
    print("Centroids: sepal length, sepal width, petal length, petal width (cm)")
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster in range(3):
        members = X[labels == cluster]
        print(f"Cluster {cluster + 1}: {len(members)} points, {centroids[cluster].round(3)}")
        ax.scatter(members[:, 0], members[:, 1], label=f"Cluster {cluster + 1}", alpha=0.75)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=140, label="Centroids")
    ax.set(xlabel="Sepal length (cm)", ylabel="Sepal width (cm)",
           title="Iris K-Means: K=3, all four features (2D projection)")
    ax.legend()
    fig.tight_layout()
    output = base / "prog8_clusters.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {output}")
