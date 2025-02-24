import os
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import importlib

def read_csv(dataset_path):
    
    current_path = os.getcwd()
    aux_curr_path = current_path
    project_path = aux_curr_path.replace('/notebooks', '')
    dataset_path = os.path.join(project_path, dataset_path)
    return dataset_path



def col_filter(input_df, col_to_filt, filt):
    filtered_df = input_df[input_df[col_to_filt] == filt] 
    return filtered_df


def pairs_x(df_col_list, int_list):
    
    list_index = []
    for i in int_list:
        for j in i:
            for k,data in enumerate(df_col_list):
                if j == data:
                    list_index.append(k)
    
    list_x = []  

    for i in range(0, len(list_index), 2):
    
        if i + 1 < len(list_index):
            list_x.append([list_index[i], list_index[i + 1]])
    
    return list_x

def var_comb(list_col):
    x_comb = [list(pair) for pair in combinations(list_col,2)]
    
    return x_comb

def get_x(x_pairs, in_df):
    X_out = []
    aux_x = []
    
    for i in x_pairs:
        aux_x = in_df.iloc[:,i].values
        X_out.append(aux_x)
    
    return X_out
    

def wss_list(x):
    
    wcss =[]
    wcss_main = []

    for j in range(len(x)): 
        for i in range(1,11):
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
            kmeans.fit(x[j])
            wcss.append(kmeans.inertia_)
        
        wcss_main.append(wcss)
        wcss = []
    return wcss_main


def plt_elbow(data_list,var_list):
    num_rows = 7
    num_cols = 3
    
    
    num_plots = len(data_list)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(9,12))
    axs = axs.flatten()
    
    for i, data in enumerate(data_list):
        axs[i].plot(data)
        axs[i].set_title(var_list[i])
        
    # Eliminar ejes vacíos si hay más subplots de los necesarios
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])  # Elimina los subplots vacíos
        
    plt.tight_layout()  # Ajusta el diseño para que no se sobrepongan
    plt.show()

def multi_plot_clusters(num_clusters, x, y, x_pairs_names):
    
    fig, axes = plt.subplots(7, 3, figsize=(25, 12))  
    axes = axes.flatten() 
    
    
    preset_colors = {
        3: ["green", "red", "blue"],
        4: ["green", "red", "blue", "black"],
        5: ["green", "red", "blue", "black", "silver"],
        6: ["green", "red", "blue", "black", "silver", "brown"]
    }
    
   
    colors = preset_colors.get(num_clusters, sns.color_palette("husl", num_clusters))

    for i in range(len(x)):  
        ax = axes[i]  

        for j in range(num_clusters):
            ax.scatter(x[i][y[i] == j, 0], x[i][y[i] == j, 1], 
                       s=50, color=colors[j], label=f"Cluster {j+1}")

        ax.set_xlim(0, 130)
        
        
        title = f"Clusters in {x_pairs_names[i]}"
        ax.set_title(title)
        
        ax.legend()

    plt.tight_layout()  
    plt.show()





def plot_agglomerative_clustering(pairs, df, n_clusters=5):
    """
    Performs agglomerative hierarchical clustering on the specified columns and generates a plot of the results.

    Parameters:
    - pairs: list of tuples, each tuple contains two column indices to be used for clustering.
    - df: pandas DataFrame containing the data.
    - n_clusters: the number of clusters to be desired (default is 5).
    """
    
    # Create a grid of subplots with 6 rows and 4 columns (adjustable)
    num_plots = len(pairs)
    rows = 6
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(25, 4 * rows))  # Adjust the figure size
    
    # Flatten the matrix of axes to make it easier to access
    axes = axes.flatten()
    
    for i, pair in enumerate(pairs):
        # Extract the columns corresponding to the indices in the pair
        X = df.iloc[:, list(pair)].values
        col_names = df.columns[list(pair)]  # Get the column names
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform agglomerative hierarchical clustering
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        Y = agg_clustering.fit_predict(X_scaled)
        
        # Get the axis corresponding to this subplot
        ax = axes[i]
        
        # Plot the points of each cluster
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=Y, cmap='viridis', marker='o')
        
        # Add title and labels with the column names
        ax.set_title(f'Agglomerative Clustering \n ({col_names[0]} & {col_names[1]})')
        ax.set_xlabel(f'{col_names[0]}')
        ax.set_ylabel(f'{col_names[1]}')
        
        # Add color bar
        fig.colorbar(scatter, ax=ax, label='Cluster Label')
    
    # Final adjustment of the plot layout
    plt.tight_layout()
    plt.show()


    
def plot_dbscan_clustering(pairs, df, eps=0.9, min_samples=4):
    """
    Performs DBSCAN clustering on the specified columns and generates a plot of the results.

    Parameters:
    - pairs: list of tuples, each tuple contains two column indices to be used for clustering.
    - df: pandas DataFrame containing the data.
    - eps: maximum distance between two samples for them to be considered as in the same neighborhood (default is 0.015).
    - min_samples: number of samples in a neighborhood for a point to be considered as a core point (default is 3).
    """
    
    # Create a grid of subplots with 6 rows and 4 columns (adjustable)
    num_plots = len(pairs)
    rows = 6
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(25, 4 * rows))  # Adjust the figure size
    
    # Flatten the matrix of axes to make it easier to access
    axes = axes.flatten()
    
    for i, pair in enumerate(pairs):
        # Extract the columns corresponding to the indices in the pair
        X = df.iloc[:, list(pair)].values
        col_names = df.columns[list(pair)]  # Get the column names
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        Y = dbscan.fit_predict(X_scaled)
        
        # Get the axis corresponding to this subplot
        ax = axes[i]
        
        # Plot the points of each cluster, where -1 is treated as noise
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=Y, cmap='viridis', marker='o')
        
        # Add title and labels with the column names
        ax.set_title(f'DBSCAN Clustering\n ({col_names[0]} & {col_names[1]})')
        ax.set_xlabel(f'{col_names[0]}')
        ax.set_ylabel(f'{col_names[1]}')
        
        # Add color bar
        fig.colorbar(scatter, ax=ax, label='Cluster Label')
    
    # Final adjustment of the plot layout
    plt.tight_layout()
    plt.show()


def plot_gmm_clustering(pairs, df, n_clusters=5):
    """
    Uses Gaussian Mixture Model (GMM) to group data points based on the provided column pairs 
    and displays the results in multiple scatter plots.
    """

    # Determine layout for displaying multiple graphs
    num_plots = len(pairs)
    rows = 6
    cols = 4
    total_plots = rows * cols  # Max plots that can be displayed
    fig, axes = plt.subplots(rows, cols, figsize=(25, 4 * rows))  # Set up figure size
    
    # Convert axes into a linear list for easy iteration
    axes = axes.flatten()

    valid_plot_count = 0  # Tracks the number of generated plots

    for pair in pairs:
        # Ensure selected column indices exist in the dataset
        if max(pair) >= len(df.columns):
            print(f"Skipping pair {pair} - column index out of range.")
            continue  # Ignore invalid columns

        # Retrieve selected columns
        X = df.iloc[:, list(pair)].values
        col_names = df.columns[list(pair)]  # Store column names
        
        # Normalize data for better clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
        Y = gmm.fit_predict(X_scaled)
        
        # Ensure we don't exceed available plotting space
        if valid_plot_count >= total_plots:
            break  # Stop adding more plots if max reached
        
        # Select next available subplot position
        ax = axes[valid_plot_count]
        
        # Generate scatter plot with cluster assignments
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=Y, cmap='viridis', marker='o')
        
        # Set up plot labels
        ax.set_title(f'GMM Clustering \n ({col_names[0]} & {col_names[1]})')
        ax.set_xlabel(f'{col_names[0]}')
        ax.set_ylabel(f'{col_names[1]}')
        
        # Attach color legend to represent clusters
        fig.colorbar(scatter, ax=ax, label='Cluster Label')

        # Increment plot counter
        valid_plot_count += 1

    # Hide empty subplots if fewer plots were generated than available space
    for j in range(valid_plot_count, total_plots):
        axes[j].set_visible(False)

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()
    plt.show()
