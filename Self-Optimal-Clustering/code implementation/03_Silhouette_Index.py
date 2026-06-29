from sklearn import metrics
def calculate_silhouette(X, labels, num_clusters):
    Labels =labels.astype(int)
    # Sample-wise Silhouette values
    sample_si = metrics.silhouette_samples(
        X.astype(np.float64),
        Labels,
        metric="sqeuclidean"
    )

    # Cluster-wise Silhouette Index
    S_m = np.zeros(num_clusters)

    for cluster in range(num_clusters):

        cluster_scores = sample_si[Labels == cluster]

        if len(cluster_scores) > 0:
            S_m[cluster] = np.mean(cluster_scores)

    # Global Silhouette Index
    GSI = np.mean(sample_si)

    return S_m, GSI
