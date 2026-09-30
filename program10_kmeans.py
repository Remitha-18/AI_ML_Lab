# Experiment 10
# Implementation of K-Means Clustering and Cluster Validation

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Sample data
# [Age, Spending Score]
X = np.array([
    [20, 20],
    [22, 25],
    [21, 18],
    [23, 22],
    [25, 28],
    [45, 75],
    [47, 80],
    [50, 78],
    [48, 82],
    [52, 85],
    [70, 30],
    [72, 35],
    [75, 32],
    [78, 38],
    [80, 40]
])


# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# Train the model
kmeans.fit(X)


# Get cluster labels
labels = kmeans.labels_


# Get cluster centers
centers = kmeans.cluster_centers_


# Calculate Silhouette Score
silhouette = silhouette_score(X, labels)


# Display results
print("----- K-MEANS CLUSTERING -----")

print("\nData:")
print(X)

print("\nCluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

print("\nSilhouette Score:")
print(round(silhouette, 3))


print("\nCluster Validation:")

if silhouette >= 0.5:
    print("Good clustering")
elif silhouette >= 0.25:
    print("Moderate clustering")
else:
    print("Poor clustering")


print("\nK-Means Clustering and Cluster Validation Completed Successfully!")