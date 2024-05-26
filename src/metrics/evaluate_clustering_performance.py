
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score

def evaluate_clustering_performance(data, n_clusters, n_init = 10, max_iter = 100):

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter).fit(data)
    ylabel = kmeans.labels_

    davies = davies_bouldin_score(data, ylabel)
    calinski = calinski_harabasz_score(data, ylabel)
    silhouette = silhouette_score(data, ylabel)

    print(f"Number of clusters: {n_clusters}")

    #The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters, 
    #with a higher score indicating a better fit of objects within their own cluster and greater separation from other clusters.
    print(f"Silhouette Score {silhouette}")

    #The Davies-Bouldin index is a metric for evaluating clustering algorithms where a lower score indicates 
    #clusters with high similarity within themselves and low similarity between clusters.
    print(f"Davies-Bouldin Index {davies}")

    #The Calinski-Harabasz index, also known as the variance ratio criterion, is a measure of cluster validity, 
    #where a higher score indicates better-defined clusters characterized by tight cohesion within clusters and good separation between clusters.
    print(f"Calinski Harabasz Score {calinski}")


    return silhouette, davies, calinski