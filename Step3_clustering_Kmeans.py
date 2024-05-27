import pandas as pd
from sklearn.cluster import KMeans

# Load the data from the Excel file
r_peaks_data = pd.read_excel("r_peaks_data_output.xlsx")

# Assuming all columns in the DataFrame are features for clustering
# You may need to select specific columns based on your data
columns_to_include = r_peaks_data.columns[:-1]  # Exclude the last column
X = r_peaks_data[columns_to_include].values

# Specify the number of clusters
n_clusters = 2 # Adjust as needed

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(X)

# Assign cluster labels starting from 0
cluster_labels = range(n_clusters)

# Map cluster assignments to cluster labels
cluster_labels_map = {cluster: label for cluster, label in zip(sorted(set(cluster_assignments)), cluster_labels)}

# Add cluster labels to the DataFrame
r_peaks_data['Cluster'] = [cluster_labels_map[cluster] for cluster in cluster_assignments]

# Save the DataFrame with cluster labels to a new Excel file
r_peaks_data.to_excel('r_peaks_data_with_clusters1.xlsx', index=False)






# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# # Load the data from the Excel file
# r_peaks_data = pd.read_excel("r_peaks_data_with_clusters.xlsx")

# # Assuming all columns in the DataFrame are features for clustering
# # You may need to select specific columns based on your data
# X = r_peaks_data.values

# # Specify the number of clusters
# n_clusters = 2  # Adjust as needed

# # Apply KMeans clustering
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# cluster_assignments = kmeans.fit_predict(X)

# # Get cluster centers
# cluster_centers = kmeans.cluster_centers_

# # Plot the clusters
# plt.figure(figsize=(8, 6))

# # Plot data points
# plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', s=50, alpha=0.5, label='Data Points')

# # Plot cluster centers
# plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')

# plt.title('K-Means Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.grid(True)
# plt.show()
