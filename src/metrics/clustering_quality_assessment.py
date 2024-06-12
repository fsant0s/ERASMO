import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, adjusted_rand_score, homogeneity_score, davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import logging

def __clusteringmetrics(data, name, y_pred, y_true = None):

    print(f"Calculating internal metrics for {name}")

    # Calculate internal metrics
    davies = davies_bouldin_score(data, y_pred)  # Lower score is better, indicates well-separated clusters
    calinski = calinski_harabasz_score(data, y_pred)  # Higher score is better, indicates tight clusters
    silhouette = silhouette_score(data, y_pred)  # Higher score is better, indicates well-separated and cohesive clusters


    # Calculate external metrics
    f1s, ari, hs = None, None, None
    if y_true is not None:
        print(f"Calculating external metrics for {name}")
        f1s = f1_score(y_true, y_pred, average='weighted')  # Balances precision and recall
        ari = adjusted_rand_score(y_true, y_pred)  # Measures the similarity of the two assignments, ignoring permutations and with chance normalization
        hs = homogeneity_score(y_true, y_pred)  # Each cluster contains only members of a single class

    results = {}
    results[name] = {
        'Silhouette Score': "{:.2f}".format(silhouette),
        'Davies-Bouldin Index': "{:.2f}".format(davies),
        'Calinski Harabasz Score': "{:.2f}".format(calinski),
        'Weighted F1-Score': "{:.2f}".format(f1s) if f1s is not None else None,
        'Adjusted Rand Index': "{:.2f}".format(ari) if ari is not None else None,
        'Homogeneity Score': "{:.2f}".format(hs) if hs is not None else None
    }

    return results

def __align_labels_with_max(y_true, y_pred):
    print("__align_labels_with_max")
    labels_unique = np.unique(y_pred)
    new_labels = np.zeros_like(y_pred)
    
    for label in labels_unique:
        mask = y_pred == label
        dominant_label = np.bincount(y_true[mask]).argmax()
        new_labels[mask] = dominant_label
    
    return new_labels

def __align_labels(true_labels, cluster_labels):
    """
    Aligns cluster labels to true labels by finding the permutation that maximizes the overall accuracy.
    """
    print("__align_labels")
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    aligned_labels = np.zeros_like(cluster_labels)
    for i, j in zip(row_ind, col_ind):
        aligned_labels[cluster_labels == j] = i
    return aligned_labels

def clustering_quality_assessment(data, dataset_name, n_clusters, y_true = None):

    print("Starting the evaluation of clustering performance.")

    clustering_algorithms = {
        'K-Means Random': KMeans(n_clusters=n_clusters, init='random', n_init=10, random_state=0),
        'K-Means++': KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=0),
        'Agglomerative Hierarchical Clustering': AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='complete'),
        'Spectral Clustering': SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', affinity='nearest_neighbors', random_state=10)
    }

    #Initialize and configure Fuzzy C-Means using scikit-fuzzy
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data.values.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None, seed=0
    )
    labels_pred_fuzzy = np.argmax(u, axis=0)
    clustering_algorithms['Fuzzy C-Means'] = labels_pred_fuzzy
    
    all_results = {
        "n_clusters": n_clusters,
        "dataset_name": dataset_name
    }
    for name, algorithm in tqdm(clustering_algorithms.items(), desc="Clustering Algorithms"):
        print(f"Running {name}")
        if name == 'Fuzzy C-Means':
            labels_pred = labels_pred_fuzzy
        else:
            print("Training")
            labels_pred = algorithm.fit_predict(data)
        
        if y_true is None: # For dataset without true labels (banking and olist)
            labels = labels_pred 
        else:
            labels = __align_labels_with_max(y_true, labels_pred)

        # Use the metric function to calculate metrics
        results = __clusteringmetrics(data=data, name=name, y_pred=labels, y_true=y_true)
        all_results.update(results)

    return all_results
