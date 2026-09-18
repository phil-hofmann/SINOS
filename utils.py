import numpy as np
import torch

# TODO crosscheck document


def leggauss_diff(x):
    x = np.asarray(x, dtype=float)
    n = x.size

    X = x[:, None]
    dX = X - X.T

    # barycentric weights
    w = 1.0 / np.prod(dX + np.eye(n), axis=1)

    # differentiation matrix
    D = (w[None, :] / w[:, None]) / (dX + np.eye(n))
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))

    return D


def make_boundary_hhalf_matrix_numpy(X_bc, W_bc, boundary_dim=1):
    """
    Dense matrix W_half for a direct Gagliardo-type approximation of
    the squared H^{1/2}(boundary) norm:

        r.T @ W_half @ r
    """

    # Pairwise Euclidean distances
    diff = X_bc[:, None, :] - X_bc[None, :, :]
    R = np.linalg.norm(diff, axis=2)

    offdiag = ~np.eye(len(X_bc), dtype=bool)

    if np.any(R[offdiag] == 0):
        raise ValueError("Duplicate boundary points detected.")

    K = np.zeros_like(R)
    K[offdiag] = 1.0 / R[offdiag] ** (boundary_dim + 1)

    B = W_bc @ K @ W_bc
    B = 0.5 * (B + B.T)

    D = np.diag(np.sum(B, axis=1))
    L = D - B

    W_half = W_bc + 2.0 * L
    W_half = 0.5 * (W_half + W_half.T)

    return W_half


def make_boundary_hhalf_matrix_torch(X_bc, W_bc, boundary_dim=1):
    """
    Dense matrix W_half for a direct Gagliardo-type approximation of
    the squared H^{1/2}(boundary) norm:

        r.T @ W_half @ r.
    """

    N = X_bc.shape[0]

    # Pairwise distances
    R = torch.cdist(X_bc, X_bc)

    # Exact off-diagonal mask: only remove i = j
    offdiag = ~torch.eye(N, dtype=torch.bool, device=X_bc.device)

    # Check for duplicated boundary points
    if torch.any(R[offdiag] == 0):
        raise ValueError(
            "There are duplicate boundary points. "
            "The direct H^{1/2} kernel has infinite off-diagonal entries."
        )

    # Singular kernel with diagonal removed
    K = torch.zeros_like(R)
    K[offdiag] = 1.0 / R[offdiag] ** (boundary_dim + 1)

    # Weighted interaction matrix
    B = W_bc @ K @ W_bc

    # Symmetrize
    B = 0.5 * (B + B.T)

    # Graph Laplacian
    D = torch.diag(torch.sum(B, dim=1))
    L = D - B

    # Approximate squared H^{1/2} norm matrix
    W_half = W_bc + 2.0 * L
    W_half = 0.5 * (W_half + W_half.T)

    return W_half
