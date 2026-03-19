import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
@torch.no_grad()
def minibatch_kmeans_gpu_vectorized(
    x: torch.Tensor,
    n_clusters: int,
    batch_size: int = 4096,
    n_iters: int = 500,
    n_init: int = 3,
    distance: str = "euclidean",   # "euclidean" or "cosine"
    init: str = "kmeans++",        # "kmeans++" or "random"
    reassignment_ratio: float = 0.0,
    reassignment_freq: int = 20,
    final_reassign: bool = True,
    final_batch_size: int | None = None,
    seed: int | None = None,
    verbose: bool = False,
):
    """
    Vectorized GPU mini-batch k-means in PyTorch.

    Args:
        x:
            Tensor of shape (N, D), on CPU. Data will be moved to GPU in batches.
        n_clusters:
            Number of clusters K.
        batch_size:
            Mini-batch size.
        n_iters:
            Number of mini-batch update steps per restart.
        n_init:
            Number of random restarts.
        distance:
            "euclidean" or "cosine".
        init:
            "kmeans++" or "random".
        reassignment_ratio:
            Reinitialize clusters whose count is below
            reassignment_ratio * max_cluster_count.
         :
            Check for low-count clusters every this many iterations.
        final_reassign:
            If True, compute labels/inertia over the full dataset at the end.
        final_batch_size:
            Chunk size for the final full assignment.
        seed:
            Optional RNG seed.
        verbose:
            Print progress.

    Returns:
        labels:
            (N,) long tensor if final_reassign=True, else None
        centroids:
            (K, D) tensor
        inertia:
            scalar tensor if final_reassign=True, else None
        counts:
            (K,) tensor of effective online counts
    """
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, D), got {tuple(x.shape)}")
    if n_clusters <= 0 or n_clusters > x.shape[0]:
        raise ValueError("n_clusters must satisfy 1 <= n_clusters <= N")
    if distance not in {"euclidean", "cosine"}:
        raise ValueError("distance must be 'euclidean' or 'cosine'")
    if init not in {"kmeans++", "random"}:
        raise ValueError("init must be 'kmeans++' or 'random'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")
    if n_init <= 0:
        raise ValueError("n_init must be positive")

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    rng = np.random.default_rng(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device('cuda')

    # Accept numpy arrays (preferred for saving GPU memory) or torch tensors
    if isinstance(x, np.ndarray):
        x_np = x
        if x_np.dtype not in (np.float32, np.float64):
            x_np = x_np.astype(np.float32)
        dtype = torch.float32 if x_np.dtype == np.float32 else torch.float64
    else:
        x_np = x.detach().cpu().numpy()
        dtype = x.dtype

    N, D = x_np.shape


    if final_batch_size is None:
        final_batch_size = batch_size

    # Work in normalized space for cosine k-means
    x_work = x_np

    def init_random(data: np.ndarray, k: int) -> torch.Tensor:
        idx = rng.choice(data.shape[0], size=k, replace=False)
        c = torch.from_numpy(data[idx]).to(device=device, dtype=dtype)
        if distance == "cosine":
            c = F.normalize(c, dim=1, eps=1e-12)
        return c

    def init_kmeanspp(data: np.ndarray, k: int) -> torch.Tensor:
        n = data.shape[0]
        centroids = torch.empty((k, D), device=device, dtype=dtype)

        first_idx = int(rng.integers(0, n))
        centroids[0] = torch.from_numpy(data[first_idx]).to(device=device, dtype=dtype)

        if distance == "euclidean":
            closest_dist_sq = np.empty(n, dtype=np.float64 if dtype == torch.float64 else np.float32)
            for start in tqdm(range(0, n, batch_size), desc="init: find closest centroid for batches", leave=False):
                end = min(start + batch_size, n)
                chunk = torch.from_numpy(data[start:end]).to(device=device, dtype=dtype)
                dist_sq = ((chunk - centroids[0:1]) ** 2).sum(dim=1)
                closest_dist_sq[start:end] = dist_sq.cpu().numpy()
            for i in tqdm(range(1, k), desc="init: per cluster initialization", leave=False):
                probs = closest_dist_sq / closest_dist_sq.sum().clip(min=1e-12)
                next_idx = int(rng.choice(n, p=probs))
                centroids[i] = torch.from_numpy(data[next_idx]).to(device=device, dtype=dtype)
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    chunk = torch.from_numpy(data[start:end]).to(device=device, dtype=dtype)
                    dist_sq = ((chunk - centroids[i:i+1]) ** 2).sum(dim=1)
                    closest_dist_sq[start:end] = np.minimum(
                        closest_dist_sq[start:end], dist_sq.cpu().numpy()
                    )
        else:
            closest_dist = np.empty(n, dtype=np.float64 if dtype == torch.float64 else np.float32)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                chunk = torch.from_numpy(data[start:end]).to(device=device, dtype=dtype)
                sims = chunk @ centroids[0:1].T
                closest_dist[start:end] = (1.0 - sims.squeeze(1)).cpu().numpy()
            for i in range(1, k):
                probs = closest_dist / closest_dist.sum().clip(min=1e-12)
                next_idx = int(rng.choice(n, p=probs))
                centroids[i] = torch.from_numpy(data[next_idx]).to(device=device, dtype=dtype)
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    chunk = torch.from_numpy(data[start:end]).to(device=device, dtype=dtype)
                    sims = chunk @ centroids[i:i+1].T
                    dist = (1.0 - sims.squeeze(1)).cpu().numpy()
                    closest_dist[start:end] = np.minimum(closest_dist[start:end], dist)

        if distance == "cosine":
            centroids = F.normalize(centroids, dim=1, eps=1e-12)
        return centroids

    def assign_batch(xb: torch.Tensor, centroids: torch.Tensor):
        """
        Returns:
            labels: (B,)
            vals:   (B,) nearest squared distance (euclidean)
                    or cosine distance = 1 - similarity (cosine)
        """
        if distance == "cosine":
            xb = F.normalize(xb, dim=1, eps=1e-12)
        if distance == "euclidean":
            xb_sq = (xb * xb).sum(dim=1, keepdim=True)              # (B, 1)
            c_sq = (centroids * centroids).sum(dim=1).unsqueeze(0) # (1, K)
            dists = xb_sq + c_sq - 2.0 * (xb @ centroids.T)        # (B, K)
            dists.clamp_min_(0.0)
            vals, labels = dists.min(dim=1)
        else:
            sims = xb @ centroids.T
            vals, labels = sims.max(dim=1)
            vals = 1.0 - vals
        return labels, vals

    def full_assign(data: np.ndarray, centroids: torch.Tensor, chunk_size: int):
        labels_out = []
        inertia = torch.zeros((), device=device, dtype=dtype)
        for start in range(0, data.shape[0], chunk_size):
            end = min(start + chunk_size, data.shape[0])
            chunk = torch.from_numpy(data[start:end]).to(device=device, dtype=dtype)
            labels, vals = assign_batch(chunk, centroids)
            labels_out.append(labels)
            inertia += vals.sum()
        return torch.cat(labels_out, dim=0), inertia

    best_centroids = None
    best_counts = None
    best_labels = None
    best_score = None

    for run in range(n_init):
        if init == "random":
            centroids = init_random(x_work, n_clusters)
        else:
            centroids = init_kmeanspp(x_work, n_clusters)

        # Online counts, one scalar per cluster
        counts = torch.zeros(n_clusters, device=device, dtype=dtype)

        for it in tqdm(range(n_iters), leave=False, desc="iters"):
            batch_idx = rng.integers(0, N, size=batch_size)
            xb = torch.from_numpy(x_work[batch_idx]).to(device=device, dtype=dtype)

            # Assignment
            labels, mb_vals = assign_batch(xb, centroids)

            # Per-cluster batch counts: (K,)
            batch_counts = torch.bincount(labels, minlength=n_clusters).to(dtype)

            # Per-cluster batch sums: (K, D)
            batch_sums = torch.zeros((n_clusters, D), device=device, dtype=dtype)
            batch_sums.index_add_(0, labels, xb)

            # Which clusters were touched this iteration?
            touched = batch_counts > 0

            # Batch means for touched clusters
            batch_means = torch.zeros_like(batch_sums)
            batch_means[touched] = batch_sums[touched] / batch_counts[touched].unsqueeze(1)

            # Online update:
            # c_new = (1 - eta_k) * c_old + eta_k * batch_mean
            # eta_k = batch_count_k / (old_count_k + batch_count_k)
            new_counts = counts + batch_counts
            eta = torch.zeros_like(counts)
            eta[touched] = batch_counts[touched] / new_counts[touched].clamp_min(1.0)

            centroids[touched] = (
                (1.0 - eta[touched]).unsqueeze(1) * centroids[touched]
                + eta[touched].unsqueeze(1) * batch_means[touched]
            )

            counts = new_counts

            if distance == "cosine":
                centroids = F.normalize(centroids, dim=1, eps=1e-12)

            # Optional low-count cluster reassignment
            if reassignment_ratio > 0 and reassignment_freq > 0 and (it + 1) % reassignment_freq == 0:
                max_count = counts.max().clamp_min(1.0)
                low = counts < (reassignment_ratio * max_count)
                n_low = int(low.sum().item())

                if n_low > 0:
                    repl_idx = rng.integers(0, N, size=n_low)
                    centroids[low] = torch.from_numpy(x_work[repl_idx]).to(device=device, dtype=dtype)
                    counts[low] = 1.0  # reset to small nonzero count

                    if distance == "cosine":
                        centroids[low] = F.normalize(centroids[low], dim=1, eps=1e-12)

            if verbose and ((it + 1) % max(1, n_iters // 10) == 0 or it == 0):
                used = int((counts > 0).sum().item())
                print(
                    f"[init {run+1}/{n_init}] "
                    f"iter {it+1}/{n_iters} "
                    f"mb_mean_dist={float(mb_vals.mean()):.6f} "
                    f"used={used}/{n_clusters}"
                )

        if final_reassign:
            labels, inertia = full_assign(x_work, centroids, final_batch_size)
            score = inertia
        else:
            labels, inertia = None, None
            # fallback proxy if skipping full evaluation
            score = -counts.gt(0).sum().to(dtype)

        if best_score is None or score < best_score:
            best_score = score.clone() if torch.is_tensor(score) else score
            best_centroids = centroids.clone()
            best_counts = counts.clone()
            best_labels = labels.clone() if labels is not None else None
            best_inertia = inertia.clone() if inertia is not None else None

    return best_labels, best_centroids, best_inertia, best_counts