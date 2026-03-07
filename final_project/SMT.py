from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm

from scipy.sparse import csr_matrix
from scipy.linalg import eigh


def extract_mel_and_mfcc(
    audio_path: str | Path,
    n_mels: int = 40,
    n_mfcc: int = 20,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> dict[str, np.ndarray | int]:
    """
    Load one audio file and extract mel-spectrogram + MFCC using file-native sample rate.

    Returns:
        dict with keys:
            - sr: sampling rate (int)
            - win_length: analysis window length in samples
            - hop_length: hop length in samples
            - mel_spectrogram: power mel spectrogram [n_mels, n_frames]
            - log_mel_db: log-mel spectrogram in dB [n_mels, n_frames]
            - mfcc: MFCC features [n_mfcc, n_frames]
    """
    y, sr = librosa.load(str(audio_path), sr=None)

    win_length = max(1, int(round(sr * (window_ms / 1000.0))))
    hop_length = max(1, int(round(sr * (hop_ms / 1000.0))))
    n_fft = win_length

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
    )

    log_mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    mfcc = librosa.feature.mfcc(
        S=log_mel_db,
        n_mfcc=n_mfcc,
    )

    return {
        "sr": sr,
        "win_length": win_length,
        "hop_length": hop_length,
        "mel_spectrogram": mel_spectrogram,
        "log_mel_db": log_mel_db,
        "mfcc": mfcc,
    }

def extract_features_from_files(
    audio_paths: Iterable[str | Path],
    n_mels: int = 40,
    n_mfcc: int = 20,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Extract mel-spectrogram and MFCC for multiple audio files."""
    results: dict[str, dict[str, np.ndarray | int]] = {}
    for path in audio_paths:
        path_str = str(path)
        results[path_str] = extract_mel_and_mfcc(
            audio_path=path,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            window_ms=window_ms,
            hop_ms=hop_ms,
        )
    return results

def extract_time_patches(
    features: np.ndarray,
    sr: int,
    hop_length: int,
    patch_ms: float = 300.0,
    patch_hop_frames: int = 1,
) -> np.ndarray:
    """
    Extract all time patches from a feature matrix.

    Args:
        features: 2D array [n_features, n_frames] (e.g., MFCC or mel-spectrogram).
        sr: Sampling rate used to generate the features.
        hop_length: Hop length in samples used to generate the features.
        patch_ms: Patch duration in milliseconds.
        patch_hop_frames: Step size between consecutive patches in frames.

    Returns:
        3D array [n_patches, n_features, patch_frames].
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2D array with shape [n_features, n_frames].")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0.")
    if patch_hop_frames <= 0:
        raise ValueError("patch_hop_frames must be > 0.")

    n_features, n_frames = features.shape
    patch_frames = max(1, int(round((patch_ms / 1000.0) * sr / hop_length)))

    if n_frames < patch_frames:
        return np.empty((0, n_features, patch_frames), dtype=features.dtype)

    starts = range(0, n_frames - patch_frames + 1, patch_hop_frames)
    patches = [features[:, s : s + patch_frames] for s in starts]
    return np.stack(patches, axis=0)

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

def preprocess_patches(patches: np.ndarray):
    
    # flatten patches to (N,40*30)
    patch_vectors = patches.reshape(patches.shape[0], -1)
    print(patch_vectors.shape)
    # subtract mean from patches
    mean = np.mean(patch_vectors, axis=1)
    std = np.std(patch_vectors, axis=1)
    patch_vectors = ((patch_vectors.T - mean)/(std + 1e-6)).T
    
    # whiten patches
    #pca = PCA(whiten=True)
    #patch_vectors = pca.fit_transform(patch_vectors)   
    zcaMatrix = zca_whitening_matrix(patch_vectors.T)
    patch_vectors = np.dot(zcaMatrix, patch_vectors.T) # project X onto the ZCAMatrix
    patch_vectors = patch_vectors.T

    # normalize patches
    patch_vectors = normalize(patch_vectors, norm='l2')
    
    return patch_vectors, mean, zcaMatrix

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
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=43, n_init=10)
    kmeans.fit(patches)
    return kmeans

def spectral_decomp(labels: np.ndarray, utterance_bounds: list, n_clusters: int):
    """
    we are solving a generalized eigen value decomposition problem: Mu=λVu
    where: 
    M=ADD^TA^T (slowness matrix)
    V=(1/N)A^TA+εI (whitening / covariance constraint)

    to avoid gigantic matrices we calculate M by formulating D as a 1D laplacian and summing over the edges of the chain
    """
    
    n_samples = labels.shape[0]

    # Sparse cluster code: shape (n_samples, n_clusters)
    A = csr_matrix(
        (np.ones(n_samples, dtype=np.float32), (np.arange(n_samples), labels)),
        shape=(n_samples, n_clusters),
    )

    # M = A@D@D.T@A.T
    M = np.zeros((A.shape[1], A.shape[1]))
    for (start, end) in utterance_bounds:
        for j in tqdm(range(start, end - 1)):
            if j+1 < A.shape[0]:
                diff = (A[j+1] - A[j]).T
                M += diff@(diff.T)
    
    A = A.T # transpose A to match smt formulation

    # calculate the constraint
    n_samples = A.shape[1]
    eps = 1e-6
    V = (A @ A.T) / n_samples
    V += eps * np.eye(V.shape[0]) # regularization which is probably necessary for 1-sparse codes

    # solve the spectral decomposition
    eigvals, eigvecs = eigh(M, V)
    return eigvals, eigvecs

def plot_first_n_mel_patches(mel_patches, n_show=10, cols=5, cmap='magma'):
    """Plot the first n_show mel patches in a single figure grid."""
    import numpy as np
    import matplotlib.pyplot as plt

    mel_patches = np.asarray(mel_patches)
    if mel_patches.ndim != 3:
        raise ValueError('mel_patches must have shape [n_patches, n_mels, n_frames].')

    n_show = min(n_show, mel_patches.shape[0])
    if n_show == 0:
        raise ValueError('mel_patches is empty.')

    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3 * cols, 2.5 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    for i in range(n_show):
        ax = axes_flat[i]
        im = ax.imshow(mel_patches[i], origin='lower', aspect='auto', cmap=cmap)
        ax.set_title(f'Patch {i}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mel bin')

    for j in range(n_show, len(axes_flat)):
        axes_flat[j].axis('off')

    # fig.colorbar(im, ax=axes, location='right', fraction=0.03, pad=0.02, label='Power')
    fig.suptitle(f'First {n_show} Mel-Spectrogram Patches')
    plt.show()

def patch_multiple_utterances(mels: list[np.ndarray], sr, hop_length):
    patches = []
    utterance_bounds = []
    start = 0
    for mel in mels:
        p = extract_time_patches(
            features=mel,
            sr=sr,
            hop_length=hop_length,
            patch_ms=300.0,
            patch_hop_frames=1,
        )
        end = start + p.shape[0]
        patches.append(p)
        utterance_bounds.append((start, end))
        start = end
    patches = np.concat(patches)
    return patches, utterance_bounds