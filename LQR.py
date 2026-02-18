import numpy as np


class LQR:
    def __init__(self, H, M, C, D, R, T):
        self.H = np.array(H, dtype = float)
        self.M = np.array(M, dtype = float)
        self.C = np.array(C, dtype = float)
        self.D = np.array(D, dtype = float)
        self.R = np.array(R, dtype = float)
        self.T = float(T)

        # validate matricies
        self.n = self.H.shape[0]
        self.m = self.M.shape[1]
        self.D_inv = np.linalg.inv(self.D)

        assert self.M.shape[0] == self.n, "M rows must match H"
        assert self.C.shape == (n, n), "C wrong dimensions"
        assert self.D.shape == (m, m), "D wrong dimensions"
        assert self.R.shape == (n, n), "R wrong dimensions"

        assert np.linalg.det(self.D) != 0, "D must be invertible"
        assert self.T > 0, "T must be positive"
    
    def ricat_RHS(self, _t, S_flat):
        S = S_flat.reshape(self.n, self.n)

        Sprime = (
                -2 * self.H.T @ S + 
                S @ self.M @ self.D_inv @self.M.T @ S 
                - self.C 
        )

        return Sprime.reshape(-1)
        










