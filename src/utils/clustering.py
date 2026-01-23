#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Module

Contains clustering algorithms specifically for protein conformations, including:
- K-means clustering
- Hierarchical clustering
- DBSCAN clustering
- Cluster validation metrics
"""

import torch
from typing import List, Tuple, Dict, Optional
from .distance_utils import calculate_atom_pair_distances
from .coordinate_utils import calculate_rmsd


def kmeans_clustering_conformations(conformations: List, n_clusters: int, max_iter: int = 100, 
                                   tolerance: float = 1e-4) -> Tuple[List, List[int]]:
    """
    K-means clustering for protein conformations based on RMSD.
    
    Args:
        conformations (List): List of conformations (each with .coordinates attribute)
        n_clusters (int): Number of clusters
        max_iter (int, optional): Maximum number of iterations
        tolerance (float, optional): Convergence tolerance
        
    Returns:
        Tuple[List, List[int]]: List of cluster centroids and cluster assignments
    """
    if not conformations:
        return [], []
    
    # Convert conformations to coordinate tensors
    coordinates = torch.stack([conf.coordinates for conf in conformations])
    num_conformations = len(conformations)
    
    # Initialize centroids randomly
    centroid_indices = torch.randperm(num_conformations)[:n_clusters]
    centroids = coordinates[centroid_indices]
    
    for i in range(max_iter):
        # Calculate RMSD from each conformation to each centroid
        rmsd_matrix = torch.zeros((num_conformations, n_clusters), dtype=torch.float32)
        for j in range(num_conformations):
            for k in range(n_clusters):
                rmsd_matrix[j, k] = calculate_rmsd(coordinates[j], centroids[k])
        
        # Assign each conformation to closest centroid
        assignments = torch.argmin(rmsd_matrix, dim=1)
        
        # Calculate new centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            cluster_members = coordinates[assignments == k]
            if len(cluster_members) > 0:
                # For centroid, use the conformation closest to the mean
                mean_coords = torch.mean(cluster_members, dim=0)
                distances = torch.tensor([calculate_rmsd(mean_coords, member) for member in cluster_members])
                closest_idx = torch.argmin(distances)
                new_centroids[k] = cluster_members[closest_idx]
            else:
                # If cluster is empty, keep original centroid
                new_centroids[k] = centroids[k]
        
        # Check convergence
        centroid_shift = torch.norm(new_centroids - centroids).item()
        if centroid_shift < tolerance:
            break
        
        centroids = new_centroids
    
    # Create centroid conformations
    centroid_conformations = []
    for centroid_coords in centroids:
        centroid_conf = conformations[0].copy()
        centroid_conf.coordinates = centroid_coords
        centroid_conformations.append(centroid_conf)
    
    return centroid_conformations, assignments.tolist()


def hierarchical_clustering(conformations: List, n_clusters: int, linkage: str = 'average') -> Tuple[List, List[int]]:
    """
    Hierarchical clustering for protein conformations.
    
    Args:
        conformations (List): List of conformations (each with .coordinates attribute)
        n_clusters (int): Number of clusters
        linkage (str, optional): Linkage method ('single', 'complete', 'average')
        
    Returns:
        Tuple[List, List[int]]: List of cluster centroids and cluster assignments
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    import numpy as np
    
    if not conformations:
        return [], []
    
    # Calculate pairwise RMSD matrix
    num_conformations = len(conformations)
    rmsd_matrix = np.zeros((num_conformations, num_conformations))
    
    for i in range(num_conformations):
        for j in range(i+1, num_conformations):
            rmsd = calculate_rmsd(conformations[i].coordinates, conformations[j].coordinates)
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd
    
    # Convert to condensed distance matrix
    condensed_distances = []
    for i in range(num_conformations):
        for j in range(i+1, num_conformations):
            condensed_distances.append(rmsd_matrix[i, j])
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method=linkage)
    
    # Assign clusters
    assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    assignments = assignments - 1  # Convert to 0-based indexing
    
    # Find cluster centroids (closest to cluster mean)
    centroid_conformations = []
    coordinates = torch.stack([conf.coordinates for conf in conformations])
    
    for k in range(n_clusters):
        cluster_members = coordinates[assignments == k]
        if len(cluster_members) > 0:
            mean_coords = torch.mean(cluster_members, dim=0)
            distances = torch.tensor([calculate_rmsd(mean_coords, member) for member in cluster_members])
            closest_idx = torch.argmin(distances)
            centroid_idx = torch.where(assignments == k)[0][closest_idx]
            centroid_conformations.append(conformations[centroid_idx])
        else:
            # If cluster is empty, add a random conformation
            centroid_conformations.append(conformations[0])
    
    return centroid_conformations, assignments.tolist()


def dbscan_clustering(conformations: List, eps: float = 2.0, min_samples: int = 3) -> Tuple[List, List[int]]:
    """
    DBSCAN clustering for protein conformations.
    
    Args:
        conformations (List): List of conformations (each with .coordinates attribute)
        eps (float, optional): Maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples (int, optional): Minimum number of samples in a neighborhood for a point to be considered a core point
        
    Returns:
        Tuple[List, List[int]]: List of cluster centroids and cluster assignments
    """
    from sklearn.cluster import DBSCAN
    import numpy as np
    
    if not conformations:
        return [], []
    
    # Calculate pairwise RMSD matrix
    num_conformations = len(conformations)
    rmsd_matrix = np.zeros((num_conformations, num_conformations))
    
    for i in range(num_conformations):
        for j in range(num_conformations):
            rmsd_matrix[i, j] = calculate_rmsd(conformations[i].coordinates, conformations[j].coordinates)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    assignments = dbscan.fit_predict(rmsd_matrix)
    
    # Find unique clusters (excluding noise points with label -1)
    unique_clusters = set(assignments)
    unique_clusters.discard(-1)
    n_clusters = len(unique_clusters)
    
    # Find cluster centroids
    centroid_conformations = []
    coordinates = torch.stack([conf.coordinates for conf in conformations])
    
    for k in unique_clusters:
        cluster_members = coordinates[assignments == k]
        if len(cluster_members) > 0:
            mean_coords = torch.mean(cluster_members, dim=0)
            distances = torch.tensor([calculate_rmsd(mean_coords, member) for member in cluster_members])
            closest_idx = torch.argmin(distances)
            centroid_idx = torch.where(assignments == k)[0][closest_idx]
            centroid_conformations.append(conformations[centroid_idx])
    
    return centroid_conformations, assignments.tolist()


def calculate_silhouette_score(conformations: List, assignments: List[int]) -> float:
    """
    Calculate silhouette score for clustering quality assessment.
    
    Args:
        conformations (List): List of conformations
        assignments (List[int]): Cluster assignments
        
    Returns:
        float: Silhouette score
    """
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    if len(conformations) < 2 or len(set(assignments)) < 2:
        return 0.0
    
    # Calculate pairwise RMSD matrix
    num_conformations = len(conformations)
    rmsd_matrix = np.zeros((num_conformations, num_conformations))
    
    for i in range(num_conformations):
        for j in range(num_conformations):
            rmsd_matrix[i, j] = calculate_rmsd(conformations[i].coordinates, conformations[j].coordinates)
    
    # Calculate silhouette score
    score = silhouette_score(rmsd_matrix, assignments, metric='precomputed')
    return score


def calculate_calinski_harabasz_score(conformations: List, assignments: List[int]) -> float:
    """
    Calculate Calinski-Harabasz score for clustering quality assessment.
    
    Args:
        conformations (List): List of conformations
        assignments (List[int]): Cluster assignments
        
    Returns:
        float: Calinski-Harabasz score
    """
    from sklearn.metrics import calinski_harabasz_score
    
    if len(conformations) < 2 or len(set(assignments)) < 2:
        return 0.0
    
    # Flatten coordinates for scoring
    coordinates = torch.stack([conf.coordinates for conf in conformations])
    flattened_coordinates = coordinates.reshape(len(conformations), -1).numpy()
    
    # Calculate Calinski-Harabasz score
    score = calinski_harabasz_score(flattened_coordinates, assignments)
    return score


def calculate_davies_bouldin_score(conformations: List, assignments: List[int]) -> float:
    """
    Calculate Davies-Bouldin score for clustering quality assessment.
    
    Args:
        conformations (List): List of conformations
        assignments (List[int]): Cluster assignments
        
    Returns:
        float: Davies-Bouldin score (lower is better)
    """
    from sklearn.metrics import davies_bouldin_score
    
    if len(conformations) < 2 or len(set(assignments)) < 2:
        return float('inf')
    
    # Flatten coordinates for scoring
    coordinates = torch.stack([conf.coordinates for conf in conformations])
    flattened_coordinates = coordinates.reshape(len(conformations), -1).numpy()
    
    # Calculate Davies-Bouldin score
    score = davies_bouldin_score(flattened_coordinates, assignments)
    return score


def evaluate_clustering(conformations: List, assignments: List[int]) -> Dict[str, float]:
    """
    Evaluate clustering using multiple metrics.
    
    Args:
        conformations (List): List of conformations
        assignments (List[int]): Cluster assignments
        
    Returns:
        Dict[str, float]: Dictionary of clustering metrics
    """
    metrics = {
        'silhouette_score': calculate_silhouette_score(conformations, assignments),
        'calinski_harabasz_score': calculate_calinski_harabasz_score(conformations, assignments),
        'davies_bouldin_score': calculate_davies_bouldin_score(conformations, assignments),
        'n_clusters': len(set(assignments)),
        'n_conformations': len(conformations)
    }
    return metrics


def optimize_clustering_parameters(conformations: List, clustering_method: str = 'kmeans', 
                                  param_range: List = None) -> Dict:
    """
    Optimize clustering parameters for the best clustering quality.
    
    Args:
        conformations (List): List of conformations
        clustering_method (str, optional): Clustering method ('kmeans', 'dbscan')
        param_range (List, optional): Range of parameters to test
        
    Returns:
        Dict: Best parameters and corresponding metrics
    """
    best_score = -float('inf')
    best_params = {}
    best_metrics = {}
    
    if clustering_method == 'kmeans':
        # Test different number of clusters
        if param_range is None:
            param_range = range(2, min(10, len(conformations)))
        
        for n_clusters in param_range:
            centroids, assignments = kmeans_clustering_conformations(conformations, n_clusters)
            metrics = evaluate_clustering(conformations, assignments)
            
            # Use silhouette score for evaluation
            if metrics['silhouette_score'] > best_score:
                best_score = metrics['silhouette_score']
                best_params = {'n_clusters': n_clusters}
                best_metrics = metrics
    
    elif clustering_method == 'dbscan':
        # Test different eps values
        if param_range is None:
            param_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        for eps in param_range:
            centroids, assignments = dbscan_clustering(conformations, eps=eps)
            metrics = evaluate_clustering(conformations, assignments)
            
            # Use silhouette score for evaluation
            if metrics['silhouette_score'] > best_score:
                best_score = metrics['silhouette_score']
                best_params = {'eps': eps}
                best_metrics = metrics
    
    return {
        'best_params': best_params,
        'best_metrics': best_metrics,
        'best_score': best_score
    }
