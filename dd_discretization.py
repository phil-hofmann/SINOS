import numpy as np
from scipy.linalg import pinv, block_diag

from discretization import Discretization


class DomainDecompositionDiscretization:

    def __init__(
        self,
        polynomial_degrees,
        cells_per_dim,
        domain=None,
    ):
        self.ref = Discretization(polynomial_degrees)

        self.n = self.ref.n
        self.m = self.ref.m

        self.r = [int(r_i) for r_i in cells_per_dim]

        if len(self.r) != self.m:
            raise ValueError(
                "cells_per_dim must have the same length as polynomial_degrees."
            )

        if domain is None:
            domain = [(-1.0, 1.0)] * self.m

        if len(domain) != self.m:
            raise ValueError("domain must have the same length as polynomial_degrees.")

        self.domain = [(float(a), float(b)) for a, b in domain]

        for a, b in self.domain:
            if not a < b:
                raise ValueError("Each domain interval must satisfy a < b.")

        self.ref_shape = self.ref.shape
        self.N_ref = self.ref.N

        self.num_cells = int(np.prod(self.r))
        self.N = self.num_cells * self.N_ref

        self.shape = (self.num_cells,) + self.ref_shape

        self._init_cells()
        self._init_sinos_pinvs()

    def _init_cells(self):
        self.edges = [
            np.linspace(a, b, r_i + 1) for (a, b), r_i in zip(self.domain, self.r)
        ]

        self.cell_indices = list(np.ndindex(*self.r))

        self.cell_bounds = []
        self.cell_centers = []
        self.cell_halfwidths = []

        X_parts = [[] for _ in range(self.m)]
        w_parts = []
        cell_id_parts = []

        for c, idx in enumerate(self.cell_indices):
            bounds = [
                (self.edges[j][idx[j]], self.edges[j][idx[j] + 1])
                for j in range(self.m)
            ]

            center = np.array([(a + b) / 2.0 for a, b in bounds])
            halfwidth = np.array([(b - a) / 2.0 for a, b in bounds])

            self.cell_bounds.append(bounds)
            self.cell_centers.append(center)
            self.cell_halfwidths.append(halfwidth)

            # Shift + scale reference nodes to physical cell
            x_phys_1d = [
                center[j] + halfwidth[j] * self.ref.x[j] for j in range(self.m)
            ]

            mesh = np.meshgrid(*x_phys_1d, indexing="ij")

            for j in range(self.m):
                X_parts[j].append(mesh[j].reshape(-1, order="C"))

            # Scale reference volume weights
            jac = float(np.prod(halfwidth))
            w_parts.append(jac * self.ref.w_md)

            cell_id_parts.append(np.full(self.N_ref, c, dtype=int))

        self.X = tuple(np.concatenate(parts) for parts in X_parts)
        self.points = np.column_stack(self.X)

        self.w_md = np.concatenate(w_parts)
        self.W = np.diag(self.w_md)

        self.cell_id = np.concatenate(cell_id_parts)

        self.cell_centers = np.asarray(self.cell_centers)
        self.cell_halfwidths = np.asarray(self.cell_halfwidths)

    def _init_sinos_pinvs(self):
        """
        Reuse the reference directional pieces.

        Reference:
            H_ref = I + sum_j Hmat_parts[j]

        Physical cell:
            H_cell = I + sum_j (1 / h_j**2) Hmat_parts[j]

        Translation does not affect derivatives.
        Only the halfwidth h_j rescales derivatives.
        """

        # Because edges are made with np.linspace, all cells have the
        # same halfwidths in each dimension.
        halfwidth = self.cell_halfwidths[0]
        scale2 = 1.0 / halfwidth**2

        H_cell = self.ref.Id.copy()
        H_cell_s = self.ref.Id.copy()

        for j in range(self.m):
            H_cell += scale2[j] * self.ref.Hmat_parts[j]
            H_cell_s += scale2[j] * self.ref.Hmat_s_parts[j]

        H_cell = 0.5 * (H_cell + self.ref.W_inv @ H_cell.T @ self.ref.W)

        H_cell_s = 0.5 * (H_cell_s + self.ref.W_inv @ H_cell_s.T @ self.ref.W)

        self.Hmat_cell = H_cell
        self.Hmat_s_cell = H_cell_s

        # This is the only pinv computation.
        # Small local cell only.
        self.Hmat_cell_pinv = pinv(H_cell)
        self.Hmat_s_cell_pinv = pinv(H_cell_s)

        # Global DD pinvs are just block diagonal copies.
        self.Hmat_pinv = block_diag(*([self.Hmat_cell_pinv] * self.num_cells))

        self.Hmat_s_pinv = block_diag(*([self.Hmat_s_cell_pinv] * self.num_cells))

    def init_boundary_quadrature(self):
        if self.m != 2:
            raise NotImplementedError(
                "Boundary quadrature is currently implemented for m=2 only."
            )

        X_bc = []
        w_bc = []

        x_left, x_right = self.domain[0]
        y_bottom, y_top = self.domain[1]

        # Left / right boundary:
        # loop over cells in y-direction
        for j_cell in range(self.r[1]):
            y0 = self.edges[1][j_cell]
            y1 = self.edges[1][j_cell + 1]

            cy = 0.5 * (y0 + y1)
            hy = 0.5 * (y1 - y0)

            y_nodes = cy + hy * self.ref.x[1]
            y_weights = hy * self.ref.w[1]

            for y, w in zip(y_nodes, y_weights):
                X_bc.append([x_left, y])
                w_bc.append(w)

                X_bc.append([x_right, y])
                w_bc.append(w)

        # Bottom / top boundary:
        # loop over cells in x-direction
        for i_cell in range(self.r[0]):
            x0 = self.edges[0][i_cell]
            x1 = self.edges[0][i_cell + 1]

            cx = 0.5 * (x0 + x1)
            hx = 0.5 * (x1 - x0)

            x_nodes = cx + hx * self.ref.x[0]
            x_weights = hx * self.ref.w[0]

            for x, w in zip(x_nodes, x_weights):
                X_bc.append([x, y_bottom])
                w_bc.append(w)

                X_bc.append([x, y_top])
                w_bc.append(w)

        self.X_bc = np.asarray(X_bc, dtype=float)
        self.W_bc = np.diag(w_bc).astype(float)
