import numpy as np

from numpy.polynomial.legendre import leggauss
from scipy.linalg import pinv


class Discretization:

    def __init__(self, polynomial_degrees):
        self.n = [int(n_i) for n_i in polynomial_degrees]
        self.m = len(self.n)

        self.shape = tuple(n_i + 1 for n_i in self.n)
        self.N = int(np.prod(self.shape))

        self._init_1d_operators()
        self._init_md_operators()

    def _init_1d_operators(self):
        x_list, w_list, dx_list, dx_s_list = [], [], [], []

        for n_i in self.n:
            x_i, w_i = leggauss(n_i + 1)
            dx_i = self._leggauss_diff(x_i)
            dx_s_i = (1.0 / w_i).reshape(-1, 1) * dx_i.T * w_i

            x_list.append(x_i)
            w_list.append(w_i)
            dx_list.append(dx_i)
            dx_s_list.append(dx_s_i)

        self.x = x_list
        self.w = w_list
        self.dx = dx_list
        self.dx_s = dx_s_list

    def _init_md_operators(self):

        # Defining the grid
        self.X = tuple(np.meshgrid(*self.x, indexing="ij"))

        # Defining the quadrature weights
        self.w_md = self._kron_all(self.w)
        self.W = np.diag(self.w_md)
        self.W_inv = np.diag(1.0 / self.w_md)

        # Defining the operators for SINOS and D-SINOS
        Id_1d = [np.eye(n_i + 1) for n_i in self.n]
        self.Id = self._kron_all(Id_1d)

        self.nabla = tuple(
            self._kron_all([self.dx[i] if i == j else Id_1d[j] for j in range(self.m)])
            for i in range(self.m)
        )
        self.nabla_s = tuple(
            self._kron_all(
                [self.dx_s[i] if i == j else Id_1d[j] for j in range(self.m)]
            )
            for i in range(self.m)
        )

        self.Hmat_parts = tuple(D_s @ D for D, D_s in zip(self.nabla, self.nabla_s))
        self.Hmat_s_parts = tuple(D @ D_s for D, D_s in zip(self.nabla, self.nabla_s))

        Hmat = self.Id.copy()
        Hmat_s = self.Id.copy()

        for A, A_s in zip(self.Hmat_parts, self.Hmat_s_parts):
            Hmat += A
            Hmat_s += A_s

        self.Hmat = 0.5 * (Hmat + self.W_inv @ Hmat.T @ self.W)
        self.Hmat_s = 0.5 * (Hmat_s + self.W_inv @ Hmat_s.T @ self.W)

        self.Hmat_pinv = pinv(Hmat)
        self.Hmat_s_pinv = pinv(Hmat_s)

    def init_boundary_quadrature(self):
        if self.m != 2:
            raise NotImplementedError(
                "Boundary quadrature is currently implemented for m=2 only."
            )

        X_bc = []
        w_bc = []

        # left / right edges of reference square
        for y, w in zip(self.x[1], self.w[1]):
            X_bc.append([-1.0, y])
            w_bc.append(w)

            X_bc.append([1.0, y])
            w_bc.append(w)

        # bottom / top edges of reference square
        for x, w in zip(self.x[0], self.w[0]):
            X_bc.append([x, -1.0])
            w_bc.append(w)

            X_bc.append([x, 1.0])
            w_bc.append(w)

        self.X_bc = np.asarray(X_bc, dtype=float)
        self.W_bc = np.diag(w_bc).astype(float)

    @staticmethod
    def _kron_all(ops):
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out

    @staticmethod
    def _leggauss_diff(x):
        x = np.asarray(x, dtype=float)
        n = x.size

        X = x[:, None]
        dX = X - X.T

        bary_w = 1.0 / np.prod(dX + np.eye(n), axis=1)

        D = (bary_w[None, :] / bary_w[:, None]) / (dX + np.eye(n))
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))

        return D
