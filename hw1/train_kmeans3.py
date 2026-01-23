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
from kmeans_pytorch import kmeans


def preprocess_patches(patches):
    # flatten patches to (N,25)
    patch_vectors = patches.view(patches.shape[0], -1).float()
    
    # normalize patches using torch
    patch_vectors = torch.nn.functional.normalize(patch_vectors, p=2, dim=1)
    return patch_vectors


if __name__=="__main__":

    with gzip.open('./data/patches.pt.gz', 'rb') as f:
        patches = torch.load(f)

    # Apply k-means with K=100 on a sample of 10,000 patches
    patch_vectors = preprocess_patches(patches)

    # kmeans
    for i in tqdm(np.arange(100, 10000, 100)):
        cluster_ids_x, cluster_centers = kmeans(
            X=patch_vectors, num_clusters=i, distance='euclidean', device=torch.device('cuda:0')
        )
        with gzip.open(f'./data/cluster_centers_{i}.pt.gz', 'wb') as f:
                torch.save(cluster_centers, f)
    