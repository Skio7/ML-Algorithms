import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate some random data for clustering
np.random.seed(0)
X = np.random.randn(100, 2)

# Perform K-means clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting the clusters and centroids
colors = ['r', 'g', 'b']
for i in range(k):
    cluster = X[labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering (Using scikit-learn)')
plt.legend()
plt.show()
