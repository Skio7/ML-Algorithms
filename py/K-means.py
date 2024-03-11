import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to initialize centroids randomly
def initialize_centroids(data, k):
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_indices]
    return centroids

# Function to assign each data point to the nearest centroid
def assign_to_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        clusters[closest_centroid_index].append(point)
    return clusters

# Function to update centroids based on the mean of the points assigned to each cluster
def update_centroids(clusters):
    new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
    return np.array(new_centroids)

# Function to check convergence by comparing centroids
def has_converged(old_centroids, new_centroids):
    return (old_centroids == new_centroids).all()

# Function to perform K-means clustering
def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        old_centroids = centroids.copy()
        clusters = assign_to_clusters(data, centroids)
        centroids = update_centroids(clusters)
        if has_converged(old_centroids, centroids):
            break
    return centroids, clusters

# Generate some random data for clustering
np.random.seed(0)
X = np.random.randn(100, 2)

# Perform K-means clustering
k = 3
centroids, clusters = kmeans(X, k)

# Plotting the clusters and centroids
colors = ['r', 'g', 'b']
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering')
plt.legend()
plt.show()
