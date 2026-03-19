from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional
from FISTA import TopKSparseCoder
import torch
from typing import List, Tuple
from tqdm import tqdm

DiffOrder = Literal["first", "second", "both"]


@dataclass
class SMTStats:
    """
    Container for accumulated SMT statistics.

    Attributes
    ----------
    M : torch.Tensor
        Slowness / manifold matrix, shape (K, K)
    V : torch.Tensor
        Covariance / whitening matrix, shape (K, K)
    n_frames : int
        Total number of code frames accumulated
    n_diffs : int
        Total number of temporal difference samples accumulated
    """
    M: torch.Tensor
    V: torch.Tensor
    n_frames: int
    n_diffs: int


class DenseKSparseSMTAccumulator:
    """
    Streamed accumulator for SMT statistics using dense k-sparse codes.

    This class accumulates matrices for a generalized eigenvalue problem:

        M u = lambda V u

    where:
      - V is the code covariance / whitening matrix
      - M is the temporal slowness matrix based on first or second differences

    Notes
    -----
    - Input codes should be shape (T, K), where:
        T = number of frames / patches in one utterance
        K = code dimension / dictionary size
    - The codes can be dense tensors, even if each row is k-sparse.
    - Utterance boundaries are respected automatically because you call
      `update()` once per utterance.
    """

    def __init__(
        self,
        dictionary: torch.tensor,
        K: int, 
        batch_size: int = 500,
        enforce_positive_coefficients: bool = True,
        normalize_dictionary: bool = True,
        FISTA_Iters: int = 1000,
        diff_order: DiffOrder = "first",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        center_codes: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        dictionary : torch.tensor
            learned dictionary
        K : int
            sparseness factor
        batch_size : int
            number of utterances to process at a time
        enforce_positive_coefficients : bool
            whether to enforce positivity in sparse coefficients
        normalize_dictionary : bool
            normalize dict before calculating sparse codes
        FISTA_Iters : int
            number of iterations to run FISTA for
        diff_order : {"first", "second"}
            Whether to use first-order or second-order temporal differences.
        device : torch.device, optional
            Device for the accumulators.
        dtype : torch.dtype
            Accumulator dtype. float64 is recommended for numerical stability.
        center_codes : bool
            If True, subtract the per-utterance mean from A before accumulating.
            Usually False is a good default for initial experiments.
        """
        if diff_order not in ("first", "second"):
            raise ValueError("diff_order must be 'first' or 'second'")

        self.code_dim = dictionary.shape[0]
        self.batch_size = batch_size
        self.diff_order = diff_order
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.center_codes = center_codes
        self.k_sparse_coder = TopKSparseCoder(dictionary.to(dtype=torch.float32), 
                                              K, 
                                              FISTA_Iters, 
                                              enforce_positive_coefficients, 
                                              normalize_dictionary)

        self.M = torch.zeros((self.code_dim, self.code_dim), device=self.device, dtype=self.dtype)
        self.V = torch.zeros((self.code_dim, self.code_dim), device=self.device, dtype=self.dtype)
        self.sparse_codes_per_batch = []
        self.n_frames = 0
        self.n_diffs = 0

    @torch.no_grad()
    def update(self, batch: torch.Tensor, utterance_bounds: List[Tuple[int, int]]) -> None:
        """
        Accumulate statistics from one utterance.

        Parameters
        ----------
        A:
            Dense code matrix of shape (N, n_clusters), where each row is a code a_t.
        utterance_bounds:
            List of (start, end) index pairs. Each utterance contributes
            second-order differences for t = start+1, ..., end-2.
            `end` is exclusive.
        Raises
        ------
        ValueError
            If A does not have shape (T, K) with K == code_dim.
        """
        if batch.ndim != 2:
            raise ValueError(f"A must have shape (T, K), got {tuple(A.shape)}")
   

        if batch.shape[0] == 0:
            return

        # Move to accumulator device/dtype without modifying caller tensor.
        batch = batch.to(device=self.device, dtype=self.dtype)

        first_utterance_start = utterance_bounds[0][0]

        A = self.k_sparse_coder(batch)
        
        if A.shape[1] != self.code_dim:
            raise ValueError(
                f"A.shape[1] = {A.shape[1]} does not match code_dim = {self.code_dim}"
            )

        if self.center_codes:
            A = A - A.mean(dim=0, keepdim=True)

        # Covariance / whitening term
        # V += A^T A
        self.V.add_(A.T @ A)
        self.n_frames += A.shape[0]

        for start, end in utterance_bounds:
            start = start - first_utterance_start
            end = end - first_utterance_start
            # Need at least 3 samples to form a_{t+1} - 2a_t + a_{t-1}
            if end - start < 3:
                continue
            
            # Slowness term
            if self.diff_order == "first":
                if A.shape[0] >= 2:
                    dA = A[start+1:end] - A[start:end-1]  # (T-1, K)
                    self.M.add_(dA.T @ dA)
                    self.n_diffs += dA.shape[0]

            elif self.diff_order == "second":
                if A.shape[0] >= 3:
                    d2A = A[start+2:end] - 2.0 * A[start+1:end-1] + A[start:end-2]  # (T-2, K)
                    self.M.add_(d2A.T @ d2A)
                    self.n_diffs += d2A.shape[0]
        
        # save sparse codes to batch
        self.sparse_codes_per_batch.append(A.cpu().to_sparse_csr())
    
    @torch.no_grad()
    def batch_process_patches(self, patches: torch.tensor, utterance_bounds: List[Tuple[int, int]]) -> None:
        n = len(utterance_bounds)
        for start in tqdm(range(0, n, self.batch_size)):
            end = min(start + self.batch_size, n)
            self.update(patches[utterance_bounds[start][0]:utterance_bounds[end-1][1]], utterance_bounds[start:end])

    @torch.no_grad()
    def finalize(
        self,
        normalize: bool = True,
        eps: float = 1e-5,
        symmetrize: bool = True,
    ) -> SMTStats:
        """
        Finalize the accumulated matrices.

        Parameters
        ----------
        normalize : bool
            If True, divide V by n_frames and M by n_diffs.
        eps : float
            Ridge regularization added to V: V <- V + eps * I
        symmetrize : bool
            If True, enforce exact symmetry numerically on M and V.

        Returns
        -------
        SMTStats
            Finalized matrices and counts.
        """
        M = self.M.clone()
        V = self.V.clone()

        if normalize:
            if self.n_frames > 0:
                V /= float(self.n_frames)
            if self.n_diffs > 0:
                M /= float(self.n_diffs)

        if symmetrize:
            M = 0.5 * (M + M.T)
            V = 0.5 * (V + V.T)

        if eps > 0.0:
            V = V + eps * torch.eye(self.code_dim, device=V.device, dtype=V.dtype)

        return SMTStats(M=M, V=V, n_frames=self.n_frames, n_diffs=self.n_diffs)