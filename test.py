# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

os.environ["OMP_NUM_THREADS"] = "2"

def co2_emission(co2_data):
    # Import data
    data = pd.read_csv("co2_emmision.csv", skiprows=4)
    data = data.loc[:, ~data.columns.isin(["Country Code", "Indicator Name", "Indicator Code"])]
    data = data.dropna(axis=1, how="all")
    # Replace NaN values with 0
    data = data.fillna(0)

    # Transpose the data
    data_transpose = data.transpose()
    
    # Exclude the first column (Country Name) and convert remaining columns to numeric
    data = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    return data, data_transpose

def agricultural_land(agricultural_land_data):
    # Import data 
    data = pd.read_csv("agricultural_land_percent.csv", skiprows=4)
    data = data.loc[:, ~data.columns.isin(["Country Code", "Indicator Name", "Indicator Code"])]
    data = data.dropna(axis=1, how="all")
    # Replace NaN values with 0
    data = data.fillna(0)

    # Transpose the data
    data_transpose = data.transpose()
    
    # Exclude the first column (Country Name) and convert remaining columns to numeric
    data = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    return data, data_transpose

def kmeans_clustering(data, n_clusters=3, init_method='k-means++'):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Create and fit a KMeans model
    kmeans = KMeans(n_clusters=n_clusters, init=init_method, random_state=42)
    kmeans.fit(scaled_data)

    # Get cluster assignments and centroids
    cluster_assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Add the cluster assignments to the original dataset
    data_with_clusters = pd.DataFrame(data, columns=data.columns)
    data_with_clusters['Cluster'] = cluster_assignments
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(scaled_data, data_with_clusters['Cluster'])
    print(silhouette_avg)
     
    # Inverse transform to get the original scale
    data_with_clusters_backscaled = scaler.inverse_transform(scaled_data)

    return data_with_clusters, data_with_clusters_backscaled, centroids

co2_data = "co2_emmision.csv"
agricultural_land_data = "agricultural_land_percent.csv"
data, data_transpose = co2_emission(co2_data)
data_with_clusters, data_with_clusters_backscaled, centroids = kmeans_clustering(data, n_clusters=2, init_method='k-means++')

data.to_excel("output.xlsx", index = True)

def plot_clusters(data_with_clusters, centroids, x_column, y_column):
    # Plot the original data with clusters and centroids
    plt.figure(figsize=(10, 6))
   
    for cluster in data_with_clusters['Cluster'].unique():
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        plt.scatter(cluster_data[x_column], cluster_data[y_column], label=f'Cluster {cluster}', alpha=0.7, s=70)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=50, linewidths=2, color='black', label='Centroids')
       
    # Zoom in on a specific region of interest (customize these limits as needed)
    plt.xlim(x_min_value, x_max_value)
    plt.ylim(y_min_value, y_max_value)
    
    plt.title('Co2 Emission Clustering')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

# Adjust these values based on your data
x_min_value = -1 
x_max_value = 10
y_min_value = -2
y_max_value = 10

# Co2 emission
co2_data = "co2_emmision.csv"
data, data_transpose = co2_emission(co2_data)
data_with_clusters, data_with_clusters_backscaled, centroids = kmeans_clustering(data, n_clusters=3, init_method='k-means++')
plot_clusters(data_with_clusters, centroids, x_column=data.columns[30], y_column=data.columns[0])

def plot_clusters1(data_with_clusters, centroids, x_column, y_column):
    # Plot the original data with clusters and centroids
    plt.figure(figsize=(10, 6))
   
    for cluster in data_with_clusters['Cluster'].unique():
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        plt.scatter(cluster_data[x_column], cluster_data[y_column], label=f'Cluster {cluster}', alpha=0.7, s=70)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=50, linewidths=2, color='black', label='Centroids')
       
    # Zoom in on a specific region of interest (customize these limits as needed)
    plt.xlim(x_min_value, x_max_value)
    plt.ylim(y_min_value, y_max_value)
    
    plt.title('Agricultural Land')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

# Adjust these values based on your data
x_min_value = -1 
x_max_value = 10
y_min_value = -2
y_max_value = 10

# Agricultural Land data
agricultural_land_data = "agricultural_land_percent.csv"
agricultural_data, agricultural_data_transpose = agricultural_land(agricultural_land_data)
agricultural_data_with_clusters, agricultural_data_with_clusters_backscaled, agricultural_centroids = kmeans_clustering(agricultural_data, n_clusters=2, init_method='k-means++')
plot_clusters1(agricultural_data_with_clusters, agricultural_centroids, x_column=agricultural_data.columns[59], y_column=agricultural_data.columns[29])
