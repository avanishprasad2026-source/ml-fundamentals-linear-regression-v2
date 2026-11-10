import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Load a real dataset and drop labels (unsupervised setting)
data = load_wine()
X = data.data  # 13 numeric features, labels hidden for unsupervised task

# 2. Standardize features (essential for K-Means and PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Determine optimal k using Elbow Method & Silhouette Score
inertia = []
silhouette_scores = []
K_range = range(2, 7)

for k in K_range:
  kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
  cluster_labels = kmeans.fit_predict(X_scaled)
  inertia.append(kmeans.inertia_)
  silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters based on Silhouette Score: {optimal_k}")

# 4. Fit K-Means with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# 5. Apply PCA to reduce dataset to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
print(
    f"PCA Explained Variance Ratio (2D): {sum(explained_variance) * 100:.2f}% of"
    " original variance captured."
)

# 6. Produce a 2D scatter plot colored by K-Means cluster assignments
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolors='k', s=50
)
plt.title(
    f'K-Means Clustering (k={optimal_k}) projected via PCA',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}%)', fontweight='bold')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}%)', fontweight='bold')
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('cluster_pca_plot.png')
print("Cluster visualization saved successfully as 'cluster_pca_plot.png'!")
