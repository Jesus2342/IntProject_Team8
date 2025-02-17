import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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


def plt_elbow(knss_model, int_list):
    num_subplots = len(knss_model)
    fig, axs = plt.subplots(1, num_subplots, figsize=(20, 5))
    
    for i,data in enumerate(knss_model):
        axs[i].plot(data, marker="o")
        axs[i].set_title(int_list[i])
    
    plt.show()


def plot_clusters(num_clusters, x, y, x_pairs_names):
    
    fig, axes = plt.subplots(2, 3, figsize=(25, 12))  
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