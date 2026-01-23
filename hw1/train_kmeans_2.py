import numpy as np
import torch
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
import gzip
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import joblib
import concurrent.futures

def preprocess_patches(patches):
    # flatten patches to (N,25)
    patch_vectors = patches.view(patches.shape[0], -1).float()
    
    # normalize patches using torch
    patch_vectors = torch.nn.functional.normalize(patch_vectors, p=2, dim=1)
    return patch_vectors

def apply_kmeans_to_patches(patches, n_clusters=100, sample_size=None, batch_size=1000000, device='cuda'):
    """
    Apply mini-batch k-means clustering to 5x5 image patches using PyTorch (GPU-optimized, memory-efficient).
    
    Args:
        patches: torch tensor of shape (N, 25) where N is number of patches
        n_clusters: number of clusters (K)
        sample_size: if not None, randomly sample this many patches for clustering
        batch_size: size of mini-batches for processing
        device: device to run on ('cuda' or 'cpu')
    
    Returns:
        centroids: cluster centers as torch tensor of shape (n_clusters, 25)
    """
    patches = patches.to(device)
    
    if sample_size is not None and sample_size < len(patches):
        # Randomly sample patches
        indices = torch.randperm(len(patches))[:sample_size]
        patches = patches[indices]
    
    N, D = patches.shape
    
    # Initialize centroids randomly from data
    centroids = patches[torch.randperm(N)[:n_clusters]].clone()
    
    # For mini-batch: accumulate sums and counts
    centroid_sums = torch.zeros_like(centroids)
    centroid_counts = torch.zeros(n_clusters, device=device)
    
    num_batches = N // batch_size
    for _ in tqdm(range(300), leave=False):  # max iterations
        new_centroids = torch.zeros_like(centroids)
        for batch_idx in tqdm(range(num_batches), leave=False):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)
            batch_patches = patches[start:end]
            
            # Compute distances for batch
            distances = torch.cdist(batch_patches, centroids)  # (batch_size, K)
            
            # Assign clusters
            labels = torch.argmin(distances, dim=1)
            
            # Update sums and counts
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    centroid_sums[k] += batch_patches[mask].sum(dim=0)
                    centroid_counts[k] += mask.sum().float()
        
        # Update centroids
        mask = centroid_counts > 0
        new_centroids[mask] = centroid_sums[mask] / centroid_counts[mask].unsqueeze(1)
        
        # Reset for next iteration
        centroid_sums.zero_()
        centroid_counts.zero_()

        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    
    # Clear GPU memory
    del centroid_sums, centroid_counts, distances, labels
    torch.cuda.empty_cache()
    return centroids.cpu()

if __name__=="__main__":

    with gzip.open('./data/patches.pt.gz', 'rb') as f:
        patches = torch.load(f)

    # Apply k-means with K=100 on a sample of 10,000 patches
    patch_vectors = preprocess_patches(patches)    
    
    def process_cluster(n_clusters, patch_vectors):
        print(f"start {n_clusters}")
        centroids = apply_kmeans_to_patches(patch_vectors, n_clusters=n_clusters)
        print(f"trained {n_clusters} clusters")
        torch.save(centroids, f"kmeans_{n_clusters}.pt")

    with torch.no_grad():
        for i in tqdm(np.arange(2000,10000, 1000)):
            centroids = apply_kmeans_to_patches(patch_vectors, i)
            with gzip.open(f'./data/centroids_{i}.pt.gz', 'wb') as f:
                torch.save(centroids, f)
                del centroids
    
    # with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
    #     futures = [executor.submit(process_cluster, nc, patch_vectors) for nc in np.arange(10, 1010, 100)]
    #     for future in concurrent.futures.as_completed(futures):
    #         future.result()