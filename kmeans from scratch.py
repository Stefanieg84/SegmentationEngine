# No ML KMeans Clustering
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# Create KMeansClustering class
class KMeansClustering:
    # Define constructor
    def __init__(self, k=3):
        # k equals 3 by default
        self.k = k
        # No centroids by default
        self.centroids = None
    
    # Calculate distance between data point and all centroids
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, X, max_iterations=200):
        # Randomly initialise centroids within the boundaries of the data
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), 
                                           size=(self.k, X.shape[1]))
        
        # Iterate between data points to create clusters
        for _ in range(max_iterations):
            y = []
            # Calculate euclidean distance between data points
            for data_point in X:
                # Get list of distances
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                # Get the index of the smallest value
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            # Convert y to numpy array
            y = np.array(y)

            # Create list of indices in a cluster
            cluster_indices = []
            # Adjust centroid positions
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                # Account for empty clusters
                if len(indices) == 0:
                    # Set current centroid as new centroid
                    cluster_centers.append(self.centroids[i])
                else:
                    # Take the average position of the elements belonging to the cluster
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
            # if the max difference/change for any cluster is less than the below threshold, stop
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
                # Centroids equal new cluster centers
            else:
                self.centroids = np.array(cluster_centers)
        
        return y
    
# Let's test it
# make_blobs generates data and correct labels
data = make_blobs(n_samples=100, n_features=2, centers=3)
# We just want the data
random_points = data[0]

kmeans = KMeansClustering(k=3)
labels = kmeans.fit(random_points)

# Ensure that model can identify identical data regardless of the actual labels
# ari score of 1 means data is identical
ari = adjusted_rand_score(data[1], labels)
print("ARI=%.2f" % ari)

# Plot data points and centroids
plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
            marker="*", s=200)
plt.title("KMeans of MakeBlobs")
plt.show()
