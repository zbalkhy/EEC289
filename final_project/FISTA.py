import torch
import torch.nn.functional as F


def normalize_dictionary(D: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize dictionary rows (atoms) to unit L2 norm.

    D: (n_atoms, feat_dim)
    """
    return D / (D.norm(dim=1, keepdim=True) + eps)


def hard_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only the k largest-magnitude entries per row.

    x: (batch, n_atoms)
    returns: (batch, n_atoms)
    """
    if k <= 0:
        return torch.zeros_like(x)
    if k >= x.shape[1]:
        return x

    vals, idx = torch.topk(x.abs(), k=k, dim=1, largest=True, sorted=False)
    out = torch.zeros_like(x)
    out.scatter_(1, idx, x.gather(1, idx))
    return out


def hard_topk_nonnegative(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only the k largest positive entries per row, zero out negatives.
    Useful if you want nonnegative sparse codes.
    """
    x = x.clamp_min(0.0)

    if k <= 0:
        return torch.zeros_like(x)
    if k >= x.shape[1]:
        return x

    vals, idx = torch.topk(x, k=k, dim=1, largest=True, sorted=False)
    out = torch.zeros_like(x)
    out.scatter_(1, idx, vals)
    return out


@torch.no_grad()
def estimate_lipschitz(D: torch.Tensor, n_power_iter: int = 20) -> float:
    """
    Estimate L = ||D D^T||_2, which is the Lipschitz constant of
    grad_Z 0.5 ||X - ZD||_F^2.

    D: (n_atoms, feat_dim)
    """
    device = D.device
    n_atoms = D.shape[0]

    v = torch.randn(n_atoms, device=device)
    v = v / (v.norm() + 1e-12)

    for _ in range(n_power_iter):
        v = D @ (D.T @ v)   # equivalent to (D D^T) v
        v = v / (v.norm() + 1e-12)

    Av = D @ (D.T @ v)
    L = torch.dot(v, Av).item()
    return max(L, 1e-8)


def reconstruction_loss(X: torch.Tensor, Z: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    0.5 * ||X - ZD||_F^2 averaged over batch
    """
    recon = Z @ D
    return 0.5 * ((X - recon) ** 2).sum(dim=1).mean()


def fista_topk_sparse_code(
    X: torch.Tensor,
    D: torch.Tensor,
    k: int,
    n_iters: int = 100,
    step_size: float | None = None,
    nonnegative: bool = False,
    normalize_dict: bool = True,
    tol: float | None = 1e-5,
    return_history: bool = False,
):
    """
    Accelerated projected sparse coding with top-k hard thresholding.

    Solves approximately:
        min_Z 0.5 ||X - ZD||_F^2
        s.t. each row of Z is k-sparse

    This is FISTA-like acceleration plus hard top-k projection.
    Strictly speaking, exact k-sparsity uses a hard projection rather than
    the soft-threshold proximal operator of classical FISTA.

    Args:
        X: (batch, feat_dim) mel patches
        D: (n_atoms, feat_dim) dictionary
        k: sparsity level per sample
        n_iters: number of iterations
        step_size: optional fixed step size; if None, uses 1 / ||D D^T||_2
        nonnegative: enforce Z >= 0 if True
        normalize_dict: normalize dictionary atoms before coding
        tol: relative stopping threshold on code updates
        return_history: if True, also return loss history

    Returns:
        Z: (batch, n_atoms)
        history: dict (optional)
    """
    assert X.dim() == 2, f"X must be (batch, feat_dim), got {tuple(X.shape)}"
    assert D.dim() == 2, f"D must be (n_atoms, feat_dim), got {tuple(D.shape)}"
    assert X.shape[1] == D.shape[1], "feature dimensions must match"

    if normalize_dict:
        D = normalize_dictionary(D)

    B, Fdim = X.shape
    K = D.shape[0]

    if step_size is None:
        L = estimate_lipschitz(D)
        step_size = 1.0 / L

    # Initialize codes
    Z = torch.zeros(B, K, device=X.device, dtype=X.dtype)
    Y = Z.clone()

    t = 1.0
    loss_history = []

    for it in range(n_iters):
        Z_prev = Z

        # Gradient of 0.5 ||X - YD||^2 wrt Y:
        # grad = (Y D - X) D^T
        grad = (Y @ D - X) @ D.T

        # Gradient step
        U = Y - step_size * grad

        # Project to k-sparse set
        if nonnegative:
            Z = hard_topk_nonnegative(U, k)
        else:
            Z = hard_topk(U, k)

        # FISTA momentum
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        Y = Z + ((t - 1.0) / t_next) * (Z - Z_prev)
        t = t_next

        if return_history or tol is not None:
            loss = reconstruction_loss(X, Z, D)
            if return_history:
                loss_history.append(loss.item())

        if tol is not None:
            denom = Z_prev.norm() + 1e-8
            rel_change = (Z - Z_prev).norm() / denom
            if rel_change.item() < tol:
                break

    if return_history:
        return Z, {
            "loss": loss_history,
            "step_size": step_size,
            "n_iters_run": len(loss_history),
        }

    return Z

class TopKSparseCoder(torch.nn.Module):
    def __init__(
        self,
        dictionary: torch.Tensor,
        k: int,
        n_iters: int = 100,
        nonnegative: bool = False,
        normalize_dict: bool = True,
    ):
        super().__init__()
        self.register_buffer("dictionary", dictionary.clone())
        self.k = k
        self.n_iters = n_iters
        self.nonnegative = nonnegative
        self.normalize_dict = normalize_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = fista_topk_sparse_code(
            X=x,
            D=self.dictionary,
            k=self.k,
            n_iters=self.n_iters,
            nonnegative=self.nonnegative,
            normalize_dict=self.normalize_dict,
            tol=1e-5,
            return_history=False,
        )
        return z