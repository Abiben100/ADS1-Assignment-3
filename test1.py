# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
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

    plt.scatter(centroids[:, data_with_clusters.columns.get_loc(x_column)],
                centroids[:, data_with_clusters.columns.get_loc(y_column)],
                marker='D', s=50, linewidths=2, color='black', label='Centroids')

    plt.title('Co2 Emission Clustering')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()
    
x_column = data.columns[30]
y_column = data.columns[0]

# Adjust these values based on your data
x_min_value = data[x_column].min()
x_max_value = data[x_column].max()
y_min_value = data[y_column].min()
y_max_value = data[y_column].max()


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



def err_ranges(x, y_actual, y_predicted, covariance_matrix, confidence=0.95):
    """
    Calculate lower and upper bounds of confidence intervals.

    Parameters:
    - x: Independent variable values.
    - y_actual: Actual values of the dependent variable.
    - y_predicted: Predicted values of the dependent variable.
    - covariance_matrix: Covariance matrix from curve_fit.
    - confidence: Confidence level (default is 0.95 for 95% confidence interval).

    Returns:
    - lower_bound: Lower bound of the confidence interval.
    - upper_bound: Upper bound of the confidence interval.
    """
    dof = len(x) - len(covariance_matrix)
    t_value = scipy.stats.t.ppf((1 + confidence) / 2., dof)
    
    # Calculate residuals
    residuals = y_actual - y_predicted
    
    # Calculate Mean Squared Error (MSE)
    mse = np.sum(residuals**2) / dof
    
    # Ensure covariance_matrix is a 1D array
    covariance_matrix = np.asarray(covariance_matrix).ravel()

    # Calculate standard deviation of residuals
    std_err = np.sqrt(covariance_matrix * mse)

    # Ensure std_err has the same shape as y_predicted
    std_err = np.tile(std_err, len(y_predicted) // len(std_err) + 1)[:len(y_predicted)]

    lower_bound = y_predicted - t_value * std_err
    upper_bound = y_predicted + t_value * std_err

    return lower_bound, upper_bound


# Define the logistic function
def logistic_function(t, a, b, c):
    return a / (1 + np.exp(-(t - b) / c))

# Function to fit logistic function to data and plot results
def fit_and_plot_logistic(data, x_column, y_column, years_to_predict=20):
    # Extract a subset of the CO2 emission data for modeling
    subset_data = data[[x_column, y_column]].dropna()
    x_data = subset_data[x_column].values
    y_data = subset_data[y_column].values

    # Fit the model to the data using curve_fit
    params, covariance = curve_fit(logistic_function, x_data, y_data)

    # Extract the parameters
    a_fit, b_fit, c_fit = params

    # Use the fitted parameters to make predictions
    predicted_values = logistic_function(x_data, a_fit, b_fit, c_fit)

    # Predict values for future years
    future_years = np.arange(x_data.min(), x_data.max() + years_to_predict)
    predicted_future_values = logistic_function(future_years, a_fit, b_fit, c_fit)

    # Calculate confidence intervals using err_ranges function
    lower_bound, upper_bound = err_ranges(x_data, y_data, predicted_values, covariance)

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.scatter(x_data, y_data, label='Actual Data', color='blue')
    plt.plot(x_data, predicted_values, label='Fitted Logistic Function', color='red')
    plt.fill_between(x_data, lower_bound, upper_bound, color='pink', alpha=0.3, label='Confidence Range')

    # Plot predictions for future years
    plt.plot(future_years, predicted_future_values, label=f'Predictions for Next {years_to_predict} Years', linestyle='dashed', color='green')

    plt.title('Logistic Function Fitting with Confidence Range and Predictions')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.show()

# Co2 emission data
co2_data = "co2_emmision.csv"
data, _ = co2_emission(co2_data)

# Adjust these values based on your data
x_column = data.columns[30]
y_column = data.columns[0]

# Fit and plot logistic function
fit_and_plot_logistic(data, x_column, y_column)