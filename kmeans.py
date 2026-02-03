"""K-means clustering implementation for SOG compression using scipy."""

import numpy as np
from typing import Tuple
from scipy.spatial import KDTree


def kmeans(data: np.ndarray, k: int, iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means clustering using scipy KDTree for efficient nearest neighbor lookup.
    
    Args:
        data: Input data array (n_samples,) or (n_samples, n_features)
        k: Number of clusters
        iterations: Number of iterations
        
    Returns:
        Tuple of (centroids, labels)
        - centroids: (k,) or (k, n_features) array of cluster centers
        - labels: (n_samples,) array of cluster assignments
    """
    data = np.asarray(data, dtype=np.float32)
    n = len(data)
    
    # Handle case where we have fewer points than clusters
    if n < k:
        return data.copy(), np.arange(n, dtype=np.uint32)
    
    # Ensure 2D for consistent handling
    is_1d = data.ndim == 1
    if is_1d:
        data = data.reshape(-1, 1)
    
    n_features = data.shape[1]
    
    # Initialize centroids using quantile-based method for 1D, k-means++ style for multi-D
    if n_features == 1:
        sorted_data = np.sort(data[:, 0])
        centroids = np.zeros((k, 1), dtype=np.float32)
        for i in range(k):
            quantile = (2 * i + 1) / (2 * k)
            idx = min(int(quantile * n), n - 1)
            centroids[i, 0] = sorted_data[idx]
    else:
        # Random initialization with reproducibility
        np.random.seed(42)
        indices = np.random.choice(n, k, replace=False)
        centroids = data[indices].copy()
    
    labels = np.zeros(n, dtype=np.uint32)
    
    # For very large datasets, use batch processing
    batch_size = 50000
    
    for iteration in range(iterations):
        # Build KDTree for efficient nearest neighbor lookup
        tree = KDTree(centroids)
        
        # Assign labels in batches to reduce memory usage
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch = data[batch_start:batch_end]
            _, batch_labels = tree.query(batch)
            labels[batch_start:batch_end] = batch_labels
        
        # Update centroids using vectorized operations
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        
        # Use bincount for fast aggregation
        for j in range(n_features):
            sums = np.bincount(labels, weights=data[:, j], minlength=k)
            new_centroids[:, j] = sums
        
        counts = np.bincount(labels, minlength=k)
        
        # Handle empty clusters
        empty_mask = counts == 0
        if np.any(empty_mask):
            # Reinitialize empty clusters
            empty_indices = np.where(empty_mask)[0]
            random_points = np.random.choice(n, len(empty_indices), replace=False)
            new_centroids[empty_indices] = data[random_points]
            counts[empty_indices] = 1
        
        centroids = new_centroids / counts[:, np.newaxis]
    
    if is_1d:
        centroids = centroids.flatten()
    
    return centroids.astype(np.float32), labels


def cluster_1d(columns: dict, column_names: list, k: int = 256, iterations: int = 10) -> Tuple[np.ndarray, dict]:
    """
    Cluster multiple columns as if they were independent 1D datasets.
    
    This matches the TypeScript cluster1d function which clusters each column
    independently but using shared centroids.
    
    Args:
        columns: Dictionary of column name -> data array
        column_names: Names of columns to cluster
        k: Number of clusters (default 256)
        iterations: Number of k-means iterations
        
    Returns:
        Tuple of (centroids, labels_dict)
        - centroids: (k,) array of centroid values
        - labels_dict: Dictionary of column name -> labels array (uint8)
    """
    # Stack all column data into one 1D array
    data_list = [columns[name] for name in column_names]
    n_rows = len(data_list[0])
    
    all_data = np.concatenate(data_list)
    
    # Run k-means on concatenated data
    centroids, labels = kmeans(all_data, k, iterations)
    
    # Order centroids smallest to largest
    order = np.argsort(centroids)
    centroids_sorted = centroids[order]
    
    # Create inverse mapping for label reordering
    inv_order = np.zeros(k, dtype=np.uint32)
    inv_order[order] = np.arange(k, dtype=np.uint32)
    
    # Reorder labels and split back into columns
    labels_reordered = inv_order[labels].astype(np.uint8)
    
    labels_dict = {}
    for i, name in enumerate(column_names):
        start = i * n_rows
        end = (i + 1) * n_rows
        labels_dict[name] = labels_reordered[start:end]
    
    return centroids_sorted, labels_dict


def cluster_sh(columns: dict, sh_names: list, k: int, iterations: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster spherical harmonics data with a two-level approach.
    
    First level: cluster all SH data points into k centroids
    Second level: quantize the centroids using 256 values
    
    Args:
        columns: Dictionary of column name -> data array
        sh_names: List of SH coefficient column names
        k: Number of first-level clusters (palette size)
        iterations: Number of k-means iterations
        
    Returns:
        Tuple of (labels, centroids_quantized, codebook)
        - labels: Point -> centroid index (uint16)
        - centroids_quantized: Quantized centroid values (n_centroids, n_coeffs)
        - codebook: 256 values for the quantization
    """
    n_rows = len(columns[sh_names[0]])
    n_coeffs = len(sh_names)
    
    print(f"  Clustering {n_rows} points in {n_coeffs} dimensions into {k} clusters...")
    
    # Stack SH data: (n_rows, n_coeffs)
    sh_data = np.column_stack([columns[name] for name in sh_names])
    
    # First level clustering (reduce iterations for large datasets)
    effective_iterations = min(iterations, 3) if n_rows > 50000 else iterations
    centroids, labels = kmeans(sh_data, k, effective_iterations)
    labels = labels.astype(np.uint16)
    
    print(f"  Quantizing {k} centroids with 256-value codebook...")
    
    # Second level: quantize the centroids channel-wise
    all_centroid_values = centroids.flatten()
    codebook, _ = kmeans(all_centroid_values, 256, iterations)
    
    # Order codebook smallest to largest
    order = np.argsort(codebook)
    codebook_sorted = codebook[order].astype(np.float32)
    
    # Quantize centroids using broadcasting
    # Find nearest codebook entry for each centroid value
    diffs = np.abs(centroids.flatten()[:, np.newaxis] - codebook_sorted[np.newaxis, :])
    quant_indices = np.argmin(diffs, axis=1)
    
    centroids_quantized = quant_indices.reshape(k, n_coeffs).astype(np.uint8)
    
    return labels, centroids_quantized, codebook_sorted
