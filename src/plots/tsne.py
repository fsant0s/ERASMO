import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def plot_tsne(data, n_cluster, n_init=10, max_iter=100):
    tsne = TSNE(n_components=3, verbose=1, perplexity=200, n_iter=5000, learning_rate=200)
    X_3d = tsne.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    km = KMeans(n_clusters=n_cluster, n_init=n_init, max_iter=max_iter)
    km.fit(X_3d)

    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=plt.cm.Accent(km.labels_), s=20)

    ax.set_title('T-SNE')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.set_zlabel('Componente 3')
    plt.show()