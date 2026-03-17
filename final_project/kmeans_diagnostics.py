import numpy as np
import matplotlib.pyplot as plt


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x


def _ensure_patch_matrix(
    patches: np.ndarray,
    patch_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Returns:
        flat_patches: (N, D)
        patch_shape: (H, W)
    """
    patches = np.asarray(patches)

    if patches.ndim == 3:
        n, h, w = patches.shape
        return patches.reshape(n, h * w), (h, w)

    if patches.ndim == 2:
        if patch_shape is None:
            raise ValueError(
                "patch_shape must be provided when patches are flattened."
            )
        n, d = patches.shape
        h, w = patch_shape
        if h * w != d:
            raise ValueError(
                f"patch_shape {patch_shape} incompatible with flattened dim {d}."
            )
        return patches, patch_shape

    raise ValueError(f"Expected patches shape (N,H,W) or (N,D), got {patches.shape}")


def compute_patch_energy(
    flat_patches: np.ndarray,
    mode: str = "rms",
) -> np.ndarray:
    """
    Energy proxy per patch.
    mode:
      - 'rms': sqrt(mean(x^2))
      - 'mean_abs': mean(|x|)
      - 'mean': mean(x)
    """
    flat_patches = _ensure_2d(flat_patches)

    if mode == "rms":
        return np.sqrt(np.mean(flat_patches**2, axis=1))
    if mode == "mean_abs":
        return np.mean(np.abs(flat_patches), axis=1)
    if mode == "mean":
        return np.mean(flat_patches, axis=1)

    raise ValueError(f"Unknown energy mode: {mode}")


def atom_usage_stats(
    codes: np.ndarray,
    flat_patches: np.ndarray,
    energy_mode: str = "rms",
    eps: float = 1e-12,
) -> dict:
    """
    Computes per-atom usage summaries.

    Returns dict with:
      usage_count: how many patches activate each atom (>0)
      usage_mass: total coefficient mass for each atom
      mean_coeff_when_active
      weighted_mean_patch_energy
      weighted_low_energy_frac
      weighted_mid_energy_frac
      weighted_high_energy_frac
    """
    codes = _ensure_2d(codes)
    flat_patches = _ensure_2d(flat_patches)

    if codes.shape[0] != flat_patches.shape[0]:
        raise ValueError("codes and patches must have same number of samples")

    patch_energy = compute_patch_energy(flat_patches, mode=energy_mode)

    q_low = np.quantile(patch_energy, 0.33)
    q_high = np.quantile(patch_energy, 0.67)

    low_mask = patch_energy <= q_low
    mid_mask = (patch_energy > q_low) & (patch_energy < q_high)
    high_mask = patch_energy >= q_high

    active = codes > 0
    usage_count = active.sum(axis=0)
    usage_mass = codes.sum(axis=0)

    mean_coeff_when_active = np.divide(
        usage_mass,
        np.maximum(usage_count, 1),
        dtype=np.float64,
    )

    weighted_mean_patch_energy = (codes * patch_energy[:, None]).sum(axis=0) / (
        usage_mass + eps
    )

    weighted_low_energy_frac = (codes * low_mask[:, None]).sum(axis=0) / (usage_mass + eps)
    weighted_mid_energy_frac = (codes * mid_mask[:, None]).sum(axis=0) / (usage_mass + eps)
    weighted_high_energy_frac = (codes * high_mask[:, None]).sum(axis=0) / (usage_mass + eps)

    return {
        "patch_energy": patch_energy,
        "usage_count": usage_count,
        "usage_mass": usage_mass,
        "mean_coeff_when_active": mean_coeff_when_active,
        "weighted_mean_patch_energy": weighted_mean_patch_energy,
        "weighted_low_energy_frac": weighted_low_energy_frac,
        "weighted_mid_energy_frac": weighted_mid_energy_frac,
        "weighted_high_energy_frac": weighted_high_energy_frac,
    }


def compute_atom_shape_metrics(
    dictionary: np.ndarray,
    patch_shape: tuple[int, int],
    eps: float = 1e-12,
) -> dict:
    """
    Heuristics to flag suspicious atoms.

    Metrics:
      edge_ratio:
        fraction of total absolute energy in first/last time column
      boundary_jump_ratio:
        strongest adjacent-column jump relative to total variation
      flatness_ratio:
        std(atom) / mean(abs(atom))
      silence_like_ratio:
        low global contrast -> near-constant / dead atom
    """
    dictionary = _ensure_2d(dictionary)
    h, w = patch_shape

    if dictionary.shape[1] != h * w:
        raise ValueError("dictionary feature dimension does not match patch_shape")

    atoms = dictionary.reshape(dictionary.shape[0], h, w)

    abs_atoms = np.abs(atoms)
    total_abs = abs_atoms.sum(axis=(1, 2)) + eps

    first_last_cols = abs_atoms[:, :, 0].sum(axis=1) + abs_atoms[:, :, -1].sum(axis=1)
    edge_ratio = first_last_cols / total_abs

    col_means = atoms.mean(axis=1)  # (K, W)
    col_diffs = np.abs(np.diff(col_means, axis=1))  # (K, W-1)
    total_col_tv = col_diffs.sum(axis=1) + eps
    max_col_jump = col_diffs.max(axis=1)
    boundary_jump_ratio = max_col_jump / total_col_tv

    atom_std = atoms.std(axis=(1, 2))
    atom_mean_abs = abs_atoms.mean(axis=(1, 2)) + eps
    flatness_ratio = atom_std / atom_mean_abs

    silence_like_ratio = atom_std / (np.max(abs_atoms, axis=(1, 2)) + eps)

    return {
        "edge_ratio": edge_ratio,
        "boundary_jump_ratio": boundary_jump_ratio,
        "flatness_ratio": flatness_ratio,
        "silence_like_ratio": silence_like_ratio,
    }


def flag_suspicious_atoms(
    stats: dict,
    shape_metrics: dict,
    min_usage_count: int = 10,
    low_energy_frac_thresh: float = 0.70,
    edge_ratio_thresh: float = 0.22,
    boundary_jump_thresh: float = 0.55,
    silence_like_thresh: float = 0.08,
) -> dict[str, np.ndarray]:
    """
    Returns boolean masks for several suspicious categories.
    """
    usage_count = stats["usage_count"]
    low_frac = stats["weighted_low_energy_frac"]

    edge_ratio = shape_metrics["edge_ratio"]
    boundary_jump_ratio = shape_metrics["boundary_jump_ratio"]
    silence_like_ratio = shape_metrics["silence_like_ratio"]

    sufficiently_used = usage_count >= min_usage_count

    low_energy_atom = sufficiently_used & (low_frac >= low_energy_frac_thresh)
    edge_atom = sufficiently_used & (edge_ratio >= edge_ratio_thresh)
    boundary_atom = sufficiently_used & (boundary_jump_ratio >= boundary_jump_thresh)
    silence_like_atom = sufficiently_used & (silence_like_ratio <= silence_like_thresh)

    suspicious = low_energy_atom | edge_atom | boundary_atom | silence_like_atom

    return {
        "low_energy_atom": low_energy_atom,
        "edge_atom": edge_atom,
        "boundary_atom": boundary_atom,
        "silence_like_atom": silence_like_atom,
        "suspicious": suspicious,
    }


def print_atom_report(
    stats: dict,
    shape_metrics: dict,
    flags: dict,
    top_k: int = 30,
    sort_by: str = "usage_mass",
) -> None:
    """
    Text report for top atoms.
    """
    order = np.argsort(stats[sort_by])[::-1][:top_k]

    header = (
        f"{'atom':>5} | {'count':>7} | {'mass':>10} | {'meanE':>8} | "
        f"{'lowFrac':>7} | {'edge':>7} | {'jump':>7} | {'silent':>7} | flags"
    )
    print(header)
    print("-" * len(header))

    for k in order:
        atom_flags = []
        for name in ["low_energy_atom", "edge_atom", "boundary_atom", "silence_like_atom"]:
            if flags[name][k]:
                atom_flags.append(name.replace("_atom", ""))

        print(
            f"{k:5d} | "
            f"{stats['usage_count'][k]:7d} | "
            f"{stats['usage_mass'][k]:10.4f} | "
            f"{stats['weighted_mean_patch_energy'][k]:8.4f} | "
            f"{stats['weighted_low_energy_frac'][k]:7.2f} | "
            f"{shape_metrics['edge_ratio'][k]:7.2f} | "
            f"{shape_metrics['boundary_jump_ratio'][k]:7.2f} | "
            f"{shape_metrics['silence_like_ratio'][k]:7.2f} | "
            f"{', '.join(atom_flags) if atom_flags else '-'}"
        )


def plot_atoms(
    dictionary: np.ndarray,
    atom_indices: np.ndarray | list[int],
    patch_shape: tuple[int, int],
    title: str = "",
    cols: int = 5,
    figsize_scale: float = 2.5,
    cmap: str = "magma",
) -> None:
    """
    Plot selected atoms.
    """
    dictionary = _ensure_2d(dictionary)
    atom_indices = np.asarray(atom_indices, dtype=int)
    h, w = patch_shape

    n = len(atom_indices)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_scale, rows * figsize_scale))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for ax, idx in zip(axes, atom_indices):
        atom = dictionary[idx].reshape(h, w)
        im = ax.imshow(atom, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(f"Atom {idx}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin")
        #plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_top_activating_patches_for_atom(
    atom_idx: int,
    patches: np.ndarray,
    codes: np.ndarray,
    patch_shape: tuple[int, int] | None = None,
    top_n: int = 20,
    cols: int = 5,
    figsize_scale: float = 2.5,
    cmap: str = "magma",
) -> None:
    """
    Shows the real input patches with largest coefficient for one atom.
    """
    flat_patches, patch_shape = _ensure_patch_matrix(patches, patch_shape)
    h, w = patch_shape
    codes = _ensure_2d(codes)

    coeff = codes[:, atom_idx]
    top_idx = np.argsort(coeff)[::-1][:top_n]

    rows = int(np.ceil(top_n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_scale, rows * figsize_scale))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[top_n:]:
        ax.axis("off")

    for ax, idx in zip(axes, top_idx):
        patch = flat_patches[idx].reshape(h, w)
        im = ax.imshow(patch, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(f"patch {idx}\ncoef={coeff[idx]:.3f}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Top {top_n} activating real patches for atom {atom_idx}")
    fig.tight_layout()
    plt.show()


def plot_atom_vs_energy(
    stats: dict,
    atom_indices: np.ndarray | list[int] | None = None,
    sort_by: str = "usage_mass",
    top_k: int = 50,
) -> None:
    """
    Scatter plot of usage vs weighted mean energy.
    """
    usage_mass = stats["usage_mass"]
    mean_energy = stats["weighted_mean_patch_energy"]
    low_frac = stats["weighted_low_energy_frac"]

    if atom_indices is None:
        atom_indices = np.argsort(usage_mass)[::-1][:top_k]
    atom_indices = np.asarray(atom_indices, dtype=int)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        usage_mass[atom_indices],
        mean_energy[atom_indices],
        c=low_frac[atom_indices],
    )
    plt.xlabel("Atom usage mass")
    plt.ylabel("Weighted mean patch energy")
    plt.title("Atom usage vs patch energy")
    cbar = plt.colorbar(sc)
    cbar.set_label("Weighted low-energy fraction")

    for idx in atom_indices:
        plt.annotate(str(idx), (usage_mass[idx], mean_energy[idx]), fontsize=8)

    plt.tight_layout()
    plt.show()


def run_dictionary_diagnostics(
    patches: np.ndarray,
    dictionary: np.ndarray,
    codes: np.ndarray,
    patch_shape: tuple[int, int] | None = None,
    energy_mode: str = "rms",
    top_k_report: int = 30,
    top_k_plot: int = 25,
) -> dict:
    """
    Full diagnostic pass.

    Returns all computed diagnostic objects.
    """
    flat_patches, patch_shape = _ensure_patch_matrix(patches, patch_shape)
    dictionary = _ensure_2d(dictionary)
    codes = _ensure_2d(codes)

    if dictionary.shape[1] != flat_patches.shape[1]:
        raise ValueError(
            f"Dictionary dim {dictionary.shape[1]} != patch dim {flat_patches.shape[1]}"
        )
    if codes.shape != (flat_patches.shape[0], dictionary.shape[0]):
        raise ValueError(
            f"codes shape {codes.shape} should be (N,K)=({flat_patches.shape[0]},{dictionary.shape[0]})"
        )

    stats = atom_usage_stats(codes, flat_patches, energy_mode=energy_mode)
    shape_metrics = compute_atom_shape_metrics(dictionary, patch_shape)
    flags = flag_suspicious_atoms(stats, shape_metrics)

    print("\nTop atoms by usage mass:\n")
    print_atom_report(stats, shape_metrics, flags, top_k=top_k_report, sort_by="usage_mass")

    print("\nFlag counts:")
    for key, mask in flags.items():
        print(f"  {key}: {int(mask.sum())}")

    top_used = np.argsort(stats["usage_mass"])[::-1][:top_k_plot]
    plot_atoms(
        dictionary,
        top_used,
        patch_shape,
        title=f"Top {top_k_plot} atoms by usage mass",
        cols=10,
    )

    suspicious_used = np.where(flags["suspicious"])[0]
    suspicious_used = suspicious_used[np.argsort(stats["usage_mass"][suspicious_used])[::-1]]

    if len(suspicious_used) > 0:
        plot_atoms(
            dictionary,
            suspicious_used[:min(20, len(suspicious_used))],
            patch_shape,
            title="Most-used suspicious atoms",
            cols=5,
        )

    plot_atom_vs_energy(stats, top_k=50)

    return {
        "patch_shape": patch_shape,
        "stats": stats,
        "shape_metrics": shape_metrics,
        "flags": flags,
    }