# K-Means Clustering of NYC Green Taxi Trip Data
# Dataset Source: AWS Registry of Open Data
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Dataset

# Load the parquet file
df = pd.read_parquet("green_tripdata_2026-08.parquet")

# Preview dataset
print("First 5 rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())


# Feature Selection

# Select numeric variables relevant for clustering
features = df[
    [
        "trip_distance",
        "fare_amount",
        "total_amount",
        "passenger_count"
    ]
]

# Remove missing values
features = features.dropna()

# Preview selected features
print("\nSelected Features:")
print(features.head())


# Data Scaling

# Scale data so all variables contribute equally
scaler = StandardScaler()

X = scaler.fit_transform(features)


# Determine Optimal K using Elbow Method

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    wcss.append(model.inertia_)

# Plot elbow chart
plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.show()


# Apply K-Means

# Based on the elbow analysis
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Create cluster labels
features["Cluster"] = kmeans.fit_predict(X)


# Evaluate Clustering

silhouette = silhouette_score(X, features["Cluster"])

print("\nSilhouette Score:")
print(silhouette)


# Cluster Summary

cluster_summary = features.groupby("Cluster").mean()

print("\nCluster Summary:")
print(cluster_summary.round(2))

# Visualize Clusters


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=features,
    x="trip_distance",
    y="total_amount",
    hue="Cluster",
    palette="tab10"
)

plt.title("K-Means Clustering of NYC Green Taxi Trips")

plt.show()

import unittest

class TestKMeans(unittest.TestCase):

    def test_cluster_count(self):
        self.assertEqual(kmeans.n_clusters, 5)

# Run test
unittest.main(argv=[''], exit=False)

