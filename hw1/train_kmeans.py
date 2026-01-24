import numpy as np
import torch
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
import gzip
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
import joblib
import concurrent.futures
from sklearn.decomposition import PCA

def preprocess_patches(patches):
    
    print("flatten patches")
    # flatten patches to (N,25)
    patch_vectors = patches.view(patches.shape[0], -1).numpy()
    
    print("normalize patches")
    # normalize patches
    patch_vectors = normalize(patch_vectors, norm='l2')
    
    print("whiten patches")
    # whiten patches
    pca = PCA(whiten=True)
    patch_vectors = pca.fit_transform(patch_vectors)   
    
    return patch_vectors

def apply_kmeans_to_patches(patches, n_clusters=100, sample_size=1000000):
    """
    Apply k-means clustering to 5x5 image patches.
    
    Args:
        patches: torch tensor of shape (N, 5, 5) where N is number of patches
        n_clusters: number of clusters (K)
        normalize_patches: whether to normalize patches to unit norm
        sample_size: if not None, randomly sample this many patches for clustering
    
    Returns:
        kmeans: fitted KMeans object
        cluster_centers: cluster centers reshaped back to (n_clusters, 5, 5)
    """
    if sample_size is not None and sample_size < len(patches):
        # Randomly sample patches
        indices = np.random.choice(len(patches), sample_size, replace=False)
        patches = patches[indices]
    
    # Apply k-means
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(patches)
    
    # Reshape cluster centers back to (5, 5)
    #cluster_centers = kmeans.cluster_centers_.reshape(n_clusters, 5, 5)
    
    return kmeans


if __name__=="__main__":

    with gzip.open('./data/patches.pt.gz', 'rb') as f:
        patches = torch.load(f)

    # Apply k-means with K=100 on a sample of 10,000 patches
    patch_vectors = preprocess_patches(patches)

    n_clusters = np.concat((np.array([100]),np.arange(1000,11000,1000)))
    for i in tqdm(n_clusters):
        kmeans = apply_kmeans_to_patches(patch_vectors, i, sample_size=None)
        joblib.dump(kmeans, f"kmeans_{i}.pkl")
        del kmeans