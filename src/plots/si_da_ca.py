from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_si_da_ca(data):
    X = data

    # Range of K to try
    K_range = range(2, 10)

    # Storage for metric scores
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    # Calculate metrics for each K
    for K in K_range:
        kmeans = KMeans(n_clusters=K, random_state=10)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')

    plt.subplot(1, 3, 2)
    plt.plot(K_range, davies_bouldin_scores, marker='o')
    plt.title('Davies-Bouldin Score')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Davies-Bouldin Score')

    plt.subplot(1, 3, 3)
    plt.plot(K_range, calinski_harabasz_scores, marker='o')
    plt.title('Calinski-Harabasz Score')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Calinski-Harabasz Score')

    plt.tight_layout()
    plt.show()