import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def find_optimal_k(X, k_range=range(2, 11)):
    silhouette_scores = []
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        inertias.append(kmeans.inertia_)
    
    return silhouette_scores, inertias

def apply_kmeans(X, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def get_cluster_stats(df, labels, feature_columns):
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    
    stats = df_clustered.groupby('cluster')[feature_columns].agg(['mean', 'std'])
    return stats
