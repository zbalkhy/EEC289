from __future__ import annotations

from pathlib import Path
from typing import Iterable
from itertools import product

import librosa
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm

from scipy.sparse import csr_matrix
from scipy.linalg import eigh

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from FISTA import TopKSparseCoder

import torch

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

def extract_features_from_dataset(
    ds,
    n_samples = 100,
    n_mels: int = 40,
    n_mfcc: int = 20,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Extract mel-spectrogram and MFCC for multiple audio files."""
    results: dict[str, dict[str, np.ndarray | int]] = {}
    for i, sample in enumerate(ds):
        if i >= n_samples:
            break
        y = sample[0].numpy()
        sr = sample[1]
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

        results[str(sample[3]) + "_" + str(sample[4]) + "_" + str(sample[5])] =  {
            "sr": sr,
            "win_length": win_length,
            "hop_length": hop_length,
            "mel_spectrogram": mel_spectrogram.squeeze(),
            "log_mel_db": log_mel_db.squeeze(),
            "mfcc": mfcc,
        }
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
    if len(patches.shape)>=3:
        patches = patches.reshape(patches.shape[0], -1)
    
    # subtract mean from patches
    mean = np.mean(patches, axis=0)
    std = np.std(patches, axis=0)
    patches = (patches - mean)/(std + 1e-6)
    
    # whiten patches
    #pca = PCA(whiten=True)
    #patch_vectors = pca.fit_transform(patch_vectors)   
    zcaMatrix = zca_whitening_matrix(patches.T)
    # patches = np.dot(zcaMatrix, patches.T) # project X onto the ZCAMatrix
    # patches = patches.T

    # normalize patches
    patches = normalize(patches, norm='l2')
    
    return patches, mean, zcaMatrix

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

def spectral_decomp(
    labels: np.ndarray | torch.Tensor,
    utterance_bounds: list,
    n_clusters: int,
    device: torch.device | str | None = None,
    eps: float = 1e-6,
):
    """
    Generalized eigen-decomposition for slow feature transform:
        M u = λ V u
    where
        M = A D D^T A^T (slowness)
        V = (1/N) A A^T + ε I (covariance regularized)

    labels can be np.ndarray or torch.Tensor (1D ints in [0,n_clusters)).
    utterance_bounds is list of (start,end) boundaries (inclusive start, exclusive end).

    Returns: eigvals, eigvecs as torch tensors (on `device`).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if isinstance(labels, np.ndarray):
        labels_tensor = torch.from_numpy(labels.astype(np.int64))
    elif isinstance(labels, torch.Tensor):
        labels_tensor = labels.to(dtype=torch.int64)
    else:
        raise TypeError("labels must be numpy array or torch tensor")

    labels_tensor = labels_tensor.to(device=device)
    n_samples = labels_tensor.shape[0]

    if n_samples == 0:
        raise ValueError("labels must be non-empty")

    # accumulate transitions over utterances
    edges_src = []
    edges_dst = []

    for start, end in utterance_bounds:
        if end <= start + 1:
            continue
        seg = labels_tensor[start:end]
        edges_src.append(seg[:-1])
        edges_dst.append(seg[1:])

    if len(edges_src) == 0:
        raise ValueError("No valid utterance edges found in utterance_bounds")

    src = torch.cat(edges_src, dim=0)
    dst = torch.cat(edges_dst, dim=0)

    # Diagonal contribution
    count_src = torch.bincount(src, minlength=n_clusters).to(dtype=torch.float32, device=device)
    count_dst = torch.bincount(dst, minlength=n_clusters).to(dtype=torch.float32, device=device)
    M = torch.diag(count_src + count_dst)

    # Off-diagonals from transitions
    src_onehot = F.one_hot(src, num_classes=n_clusters).to(dtype=torch.float32, device=device)
    dst_onehot = F.one_hot(dst, num_classes=n_clusters).to(dtype=torch.float32, device=device)
    cross = dst_onehot.T @ src_onehot + src_onehot.T @ dst_onehot
    M = M - cross

    # V matrix is diagonal for one-hot codes
    cluster_counts = torch.bincount(labels_tensor, minlength=n_clusters).to(dtype=torch.float32, device=device)
    V_diag = cluster_counts / n_samples + eps

    inv_sqrt = torch.diag(1.0 / torch.sqrt(V_diag))

    # transform to standard eig problem: inv_sqrt * M * inv_sqrt
    M_whitened = inv_sqrt @ M @ inv_sqrt

    eigvals, eigvecs_w = torch.linalg.eigh(M_whitened)
    eigvecs = inv_sqrt @ eigvecs_w

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

def patch_multiple_utterances(mels: list[np.ndarray], sr, hop_length, patch_length=300):
    patches = []
    utterance_bounds = []
    start = 0
    for mel in mels:
        p = extract_time_patches(
            features=mel,
            sr=sr,
            hop_length=hop_length,
            patch_ms=patch_length,
            patch_hop_frames=1,
        )
        end = start + p.shape[0]
        patches.append(p)
        utterance_bounds.append((start, end))
        start = end
    patches = np.concat(patches)
    return patches, utterance_bounds

def list_audio_files(audio_dir: Path):
    # Common audio extensions
    exts = (".wav", ".mp3", ".ogg", ".flac")
    return [p for p in sorted(audio_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts]

def calc_smoothness(data: np.ndarray):
    # assume data is NxD where N is the number of samples
    diff = np.diff(data, axis=0)
    diff_2 = np.diff(data, n=2, axis=0)

    grad_norm = np.linalg.norm(diff, ord=2, axis=1)
    grad_norm_2 = np.linalg.norm(diff_2, ord=2, axis=1)
    grad_median = np.median(grad_norm)
    grad_2_median = np.median(grad_norm_2)
    return grad_median, grad_2_median

def block_permute_rows(X, block_size, rng=None):
    """
    Permute rows of X by shuffling contiguous blocks.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_rows, ...).
    block_size : int
        Number of rows per block.
    rng : np.random.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    X_perm : np.ndarray
        Block-permuted array.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_rows = X.shape[0]
    n_blocks = n_rows // block_size

    # Trim extra rows that don't fit into a full block
    trimmed = X[:n_blocks * block_size]

    blocks = trimmed.reshape(n_blocks, block_size, *X.shape[1:])
    perm = rng.permutation(n_blocks)

    permuted = blocks[perm].reshape(-1, *X.shape[1:])

    # Append leftover rows unchanged (optional)
    if n_rows % block_size != 0:
        permuted = np.concatenate([permuted, X[n_blocks * block_size:]], axis=0)

    return permuted

def compare_smoothness_to_null(data: np.ndarray, null_dist_size: int = 1000):
    # assume data is NxD where N is the number of samples
    obs_grad_median, obs_grad_2_median = calc_smoothness(data)

    perm_grad_medians = []
    perm_grad_2_medians = []
    for i in range(null_dist_size):
        perm_data = block_permute_rows(data, 10)#data[np.random.permutation(data.shape[0]),:]
        perm_grad_median, perm_grad_2_median = calc_smoothness(perm_data)
        perm_grad_medians.append(perm_grad_median)
        perm_grad_2_medians.append(perm_grad_2_median)

    perm_grad_medians = np.array(perm_grad_medians)
    perm_grad_2_medians = np.array(perm_grad_2_medians)

    z_score_smooth = (obs_grad_median - np.mean(perm_grad_medians)) / np.std(perm_grad_medians)
    z_score_linear = (obs_grad_2_median - np.mean(perm_grad_2_medians)) / np.std(perm_grad_2_medians)
    return obs_grad_2_median, obs_grad_2_median, z_score_smooth, z_score_linear

def calc_path_efficiency(data: np.ndarray):
    diff = np.diff(data, axis=0)
    grad_norm = np.linalg.norm(diff, ord=2, axis=1)
    denom = np.sum(grad_norm)
    nom = np.linalg.norm(data[-1,:] - data[0,:], ord=2)
    return nom/denom

def calc_metrics_per_utterance(mel_patches, smt_patches, utterance_bounds):
    smt_stats = {'obs_grad_median': [], 
             'obs_grad_2_median': [],
             'z_score_smooth': [],
             'z_score_linear': [],
             'path_efficiency': []}
    mel_stats = {'obs_grad_median': [], 
                'obs_grad_2_median': [],
                'z_score_smooth': [],
                'z_score_linear': [],
                'path_efficiency': []}
    for (start, end) in tqdm(utterance_bounds, desc="metrics", leave=False):
        sample_utterance_mel = mel_patches[start:end,:]
        sample_utterance_beta = smt_patches[start:end,:]

        # calc smoothness on sensed manifold and plot
        obs_grad_median_beta, obs_grad_2_median_beta, z_score_smooth_beta, z_score_linear_beta = compare_smoothness_to_null(sample_utterance_beta)
        path_efficiency = calc_path_efficiency(sample_utterance_beta)
        
        smt_stats['path_efficiency'].append(path_efficiency)
        smt_stats['obs_grad_median'].append(obs_grad_median_beta)
        smt_stats['obs_grad_2_median'].append(obs_grad_2_median_beta)
        smt_stats['z_score_smooth'].append(z_score_smooth_beta)
        smt_stats['z_score_linear'].append(z_score_linear_beta)
        
        obs_grad_median_mel, obs_grad_2_median_mel, z_score_smooth_mel, z_score_linear_mel = compare_smoothness_to_null(sample_utterance_mel)
        path_efficiency = calc_path_efficiency(sample_utterance_mel)

        mel_stats['path_efficiency'].append(path_efficiency)
        mel_stats['obs_grad_median'].append(obs_grad_median_mel)
        mel_stats['obs_grad_2_median'].append(obs_grad_2_median_mel)
        mel_stats['z_score_smooth'].append(z_score_smooth_mel)
        mel_stats['z_score_linear'].append(z_score_linear_mel)
    return smt_stats, mel_stats

def calc_smt_on_corpus(directory: Path, patch_length: int, d: int = 128):
    audio_files = list_audio_files(directory)

    # extract audio features
    print("calc log mels")
    audio_features = extract_features_from_files(audio_files)

    # this is kinda stupid, just do this in the previous function
    mels = [clip["mel_spectrogram"] for _, clip in audio_features.items()]
    mfccs = [clip["mfcc"] for _, clip in audio_features.items()]
    log_mels = [clip["log_mel_db"] for _, clip in audio_features.items()]
    hop_length = [clip["hop_length"] for _, clip in audio_features.items()][0]
    sr = [clip["sr"] for _, clip in audio_features.items()][0]

    # patch and preprocess
    print("patch utterances")
    patches, utterance_bounds = patch_multiple_utterances(log_mels, sr, hop_length, patch_length)
    norm_patches, mean, zcaMatrix = preprocess_patches(patches)

    # kmeans for sparse coding
    print("apply kmeans")
    kmeans = apply_kmeans_to_patches(norm_patches, 2000, sample_size=None)

    # spectral decomposition for manifold learning
    labels = kmeans.labels_
    n_clusters = kmeans.n_clusters 
    print("spectral decomp")
    eigvals, eigvecs = spectral_decomp(labels, utterance_bounds, n_clusters)

    # take smallest d eigenvectors as P
    P = eigvecs[:, 1:d+1].T

    n_samples = labels.shape[0]

    # Sparse cluster code: shape (n_samples, n_clusters)
    A = csr_matrix(
        (np.ones(n_samples, dtype=np.float32), (np.arange(n_samples), labels)),
        shape=(n_samples, n_clusters),
    )

    A = torch.tensor(A.T)

    # "sense" the manifold with P
    beta = P@A

    norm_beta, _, _ = preprocess_patches(beta.T) # this switches beta to be NxD like norm_patches
    return norm_beta, norm_patches, utterance_bounds

def calc_smt_on_librispeech(librispeech_ds, patch_length: int, num_utterances =  100, d: int = 128, device: torch.device | str | None = None,):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print("calc log mels")
    audio_features = extract_features_from_dataset(librispeech_ds, num_utterances)

    # this is kinda stupid, just do this in the previous function
    mels = [clip["mel_spectrogram"] for _, clip in audio_features.items()]
    mfccs = [clip["mfcc"] for _, clip in audio_features.items()]
    log_mels = [clip["log_mel_db"] for _, clip in audio_features.items()]
    hop_length = [clip["hop_length"] for _, clip in audio_features.items()][0]
    sr = [clip["sr"] for _, clip in audio_features.items()][0]

    # patch and preprocess
    print("patch utterances")
    patches, utterance_bounds = patch_multiple_utterances(log_mels, sr, hop_length, patch_length)
    norm_patches, mean, zcaMatrix = preprocess_patches(patches)

    # kmeans for sparse coding
    print("apply kmeans")
    kmeans = apply_kmeans_to_patches(norm_patches, 2000, sample_size=None)

    # spectral decomposition for manifold learning
    labels = kmeans.labels_
    n_clusters = kmeans.n_clusters 
    print("spectral decomp")
    eigvals, eigvecs = spectral_decomp(labels, utterance_bounds, n_clusters)

    # take smallest d eigenvectors as P
    P = eigvecs[:, 1:d+1].T

    n_samples = labels.shape[0]

    # Sparse cluster code: shape (n_samples, n_clusters)
    A = csr_matrix(
        (np.ones(n_samples, dtype=np.float32), (np.arange(n_samples), labels)),
        shape=(n_samples, n_clusters),
    )

    A = A.T
    A = A.todense()
    A = torch.from_numpy(A).to(dtype=torch.float32, device=device)

    # "sense" the manifold with P
    beta = P@A
    beta = beta.cpu().numpy()

    norm_beta, _, _ = preprocess_patches(beta.T) # this switches beta to be NxD like norm_patches
    return kmeans, norm_beta, norm_patches, utterance_bounds

def grid_search_patch_size_smt(directory: Path, patch_lengths: list[int], d: int = 128):
    results = {}
    for i in tqdm(patch_lengths, desc="smt_per_patch_length"):
        print("start smt on patch_size: {}".format(i))
        smt_patches, mel_patches, utterance_bounds = calc_smt_on_corpus(directory, i, d)
        smt_stats, mel_stats = calc_metrics_per_utterance(mel_patches, smt_patches, utterance_bounds)
        results[i] = {"mel_stats": mel_stats, "smt_stats": smt_stats}
    return results

def paired_bootstrap_ci(diffs, n_boot=5000, alpha=0.05, rng=None):
    rng = np.random.default_rng(rng)
    diffs = np.asarray(diffs)
    boots = []
    n = len(diffs)
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boots.append(np.median(sample))
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return np.median(diffs), lo, hi

def plot_paired_comparison(metric_mel, metric_smt, patch_length, metric_name="Curvature z-score", alternative="less"):
    metric_mel = np.asarray(metric_mel)
    metric_smt = np.asarray(metric_smt)
    diffs = metric_smt - metric_mel

    stat, p = wilcoxon(metric_smt, metric_mel, alternative=alternative)

    med_diff, ci_lo, ci_hi = paired_bootstrap_ci(diffs, rng=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    fig.suptitle("patch_length={}ms".format(patch_length))
    # Left: paired slope plot
    x0, x1 = 0, 1
    for a, b in zip(metric_mel, metric_smt):
        axes[0].plot([x0, x1], [a, b], alpha=0.35)
    axes[0].scatter(np.full_like(metric_mel, x0), metric_mel, s=20)
    axes[0].scatter(np.full_like(metric_smt, x1), metric_smt, s=20)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Mel", "SMT"])
    axes[0].set_ylabel(metric_name)
    axes[0].set_title("Per-utterance paired values")

    # Right: paired differences
    jitter = np.random.default_rng(0).normal(0, 0.04, size=len(diffs))
    axes[1].scatter(jitter, diffs, alpha=0.5, s=20)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].errorbar(
        0.22, med_diff,
        yerr=[[med_diff - ci_lo], [ci_hi - med_diff]],
        fmt="o", capsize=4
    )
    axes[1].set_xlim(-0.15, 0.45)
    axes[1].set_xticks([])
    axes[1].set_ylabel("SMT - Mel")
    axes[1].set_title(
        f"Paired differences\nmedian={med_diff:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]\n"
        f"Wilcoxon p={p:.3g}"
    )

    plt.show()
    return stat, p, med_diff, (ci_lo, ci_hi)

def spectral_decomp_dense_torch(
    A: torch.Tensor,
    utterance_bounds,
    eps: float = 1e-6,
):
    """
    A: (N, n_clusters) dense tensor
    """
    A = A.float()
    N, n_clusters = A.shape

    M = torch.zeros((n_clusters, n_clusters), device=A.device, dtype=A.dtype)

    for start, end in utterance_bounds:
        if end - start < 2:
            continue
        D = A[start + 1:end] - A[start:end - 1]   # (T-1, n_clusters)
        M += D.T @ D

    V = (A.T @ A) / N
    V += eps * torch.eye(n_clusters, device=A.device, dtype=A.dtype)

    L = torch.linalg.cholesky(V)
    Linv_M = torch.linalg.solve_triangular(L, M, upper=False, left=True)
    C = torch.linalg.solve_triangular(
        L, Linv_M.transpose(-1, -2), upper=False, left=True
    ).transpose(-1, -2)
    C = 0.5 * (C + C.T)

    eigvals, y = torch.linalg.eigh(C)
    eigvecs = torch.linalg.solve_triangular(L.T, y, upper=True, left=True)

    return eigvals, eigvecs

def calc_smt_on_librispeech_k_sparse(
        librispeech_ds,
        k_sparse: int, 
        patch_length: int,
        n_mels: int,
        window_ms: int,
        hop_length: int,
        num_clusters: int,
        num_utterances: int = 100,
        d: int = 128
):
    audio_features = extract_features_from_dataset(librispeech_ds, 
                                                   num_utterances, 
                                                   n_mels,
                                                   window_ms=window_ms, hop_ms=hop_length)

    # this is kinda stupid, just do this in the previous function
    mels = [clip["mel_spectrogram"] for _, clip in audio_features.items()]
    mfccs = [clip["mfcc"] for _, clip in audio_features.items()]
    log_mels = [clip["log_mel_db"] for _, clip in audio_features.items()]
    hop_length = [clip["hop_length"] for _, clip in audio_features.items()][0]
    sr = [clip["sr"] for _, clip in audio_features.items()][0]

    # patch and preprocess
    print("patch utterances")
    patches, utterance_bounds = patch_multiple_utterances(log_mels, sr, hop_length, patch_length)
    norm_patches, mean, zcaMatrix = preprocess_patches(patches)

    # kmeans for sparse coding
    print("apply kmeans")
    kmeans = apply_kmeans_to_patches(norm_patches, num_clusters, sample_size=None)

    # k-sparse coding
    print("sparse coding")
    k_sparse_coder = TopKSparseCoder(torch.tensor(kmeans.cluster_centers_).to(dtype=torch.float32), k_sparse, 100, True, True)
    sparse_codes = k_sparse_coder(torch.tensor(norm_patches).to(dtype=torch.float32))

    print("spectral decomp")
    eigvals, eigvecs = spectral_decomp_dense_torch(sparse_codes, utterance_bounds)

    # take smallest d eigenvectors as P
    P = eigvecs[:, 1:d+1].T

    beta = P@sparse_codes.T
    print("norm smt patches")
    norm_beta, _, _ = preprocess_patches(beta.T.numpy()) # this switches beta to be NxD like norm_patches
    norm_beta = norm_beta

    return {
        "kmeans": kmeans,
        "sparse_codes": sparse_codes,
        "norm_smt": norm_beta,
        "norm_mel": norm_patches,
        "utterance_bounds": utterance_bounds,
    }

def grid_search(
    librispeech_ds,
    k_values: list[int] | None = None,
    patch_lengths: list[int] | None = None,
    n_mels_values: list[int] | None = None,
    num_clusters_values: list[int] | None = None,
    window_ms: int = 20,
    hop_ms: int = 10,
    num_utterances: int = 100,
    d: int = 128,
    compute_metrics: bool = True,
):
    """
    Run SMT k-sparse pipeline for every hyperparameter combination.

    Returns:
        dict with:
            - results: list of per-config dicts
            - errors: list of per-config failures
            - searched: total number of configs attempted
    """
    if k_values is None:
        k_values = [1, 5, 10]
    if patch_lengths is None:
        patch_lengths = [20, 50, 80, 100, 200, 300]
    if n_mels_values is None:
        n_mels_values = [10, 20, 40]
    if num_clusters_values is None:
        num_clusters_values = [2000, 4000, 6000]

    configs = list(product(k_values, patch_lengths, n_mels_values, num_clusters_values))
    results = []
    errors = []

    for k_sparse, patch_length, n_mels, num_clusters in tqdm(configs, desc="grid_search"):
        cfg = {
            "k_sparse": k_sparse,
            "patch_length": patch_length,
            "n_mels": n_mels,
            "num_clusters": num_clusters,
            "window_ms": window_ms,
            "hop_ms": hop_ms,
            "num_utterances": num_utterances,
            "d": d,
        }

        try:
            out = calc_smt_on_librispeech_k_sparse(
                librispeech_ds=librispeech_ds,
                k_sparse=k_sparse,
                patch_length=patch_length,
                n_mels=n_mels,
                window_ms=window_ms,
                hop_length=hop_ms,
                num_clusters=num_clusters,
                num_utterances=num_utterances,
                d=d,
            )

            entry = {"config": cfg}
            if compute_metrics:
                smt_stats, mel_stats = calc_metrics_per_utterance(
                    out["norm_mel"],
                    out["norm_smt"],
                    out["utterance_bounds"],
                )
                entry["smt_stats"] = smt_stats
                entry["mel_stats"] = mel_stats
            else:
                # Keep only compact outputs by default; avoid storing large models/tensors.
                entry["n_samples"] = int(out["norm_smt"].shape[0])
                entry["feature_dim"] = int(out["norm_smt"].shape[1])

            results.append(entry)
        except Exception as exc:
            errors.append({"config": cfg, "error_type": type(exc).__name__, "error": str(exc)})

    return {
        "results": results,
        "errors": errors,
        "searched": len(configs),
    }
