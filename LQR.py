import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp, trapezoid, cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Classes
# -------------------------------------------------------

class LQR:
    """Linear Quadratic Regulator Solver using Riccati ODE"""

    def __init__(self, H, M, C, D, R, T, sigma=0.5):
        self.H, self.M = np.array(H, dtype=float), np.array(M, dtype=float)
        self.C, self.D, self.R = (np.array(C, dtype=float),
                                   np.array(D, dtype=float),
                                   np.array(R, dtype=float))
        self.T, self.sigma = float(T), sigma
        self.n, self.m = self.H.shape[0], self.M.shape[1]
        self.D_inv = np.linalg.inv(self.D)

        assert self.M.shape[0] == self.n,        "M rows must match H"
        assert self.C.shape == (self.n, self.n), "C wrong dimensions"
        assert self.D.shape == (self.m, self.m), "D wrong dimensions"
        assert self.R.shape == (self.n, self.n), "R wrong dimensions"
        assert self.T > 0,                       "T must be positive"

        self.S_interp = None
        self.g_interp = None

        if np.isscalar(sigma):
            self._sigma_sq = float(sigma) ** 2
        else:
            self._sigma_sq = None

    def ricat_RHS(self, t, S_flat):
        """Right-hand side of the Riccati ODE"""
        S = S_flat.reshape(self.n, self.n)
        Sprime = -2 * self.H.T @ S + S @ self.M @ self.D_inv @ self.M.T @ S - self.C
        return Sprime.reshape(-1)

    def solve_riccati(self, time_grid):
        """Solves Riccati ODE and builds S_interp and g_interp"""
        t_eval = (time_grid.detach().numpy()
                  if torch.is_tensor(time_grid) else np.array(time_grid))
        t_span = (self.T, np.min(t_eval))

        sol = solve_ivp(
            self.ricat_RHS,
            t_span,
            self.R.flatten(),
            t_eval=np.sort(t_eval)[::-1],
            method='RK45',
            rtol=1e-9,
            atol=1e-12,
        )

        S_values = sol.y.T.reshape(-1, self.n, self.n)
        self.S_interp = interp1d(sol.t, S_values, axis=0,
                                 bounds_error=False, fill_value="extrapolate")

        t_asc = sol.t[::-1].copy()
        S_asc = S_values[::-1]

        if self._sigma_sq is not None:
            trace_vals = self._sigma_sq * np.trace(S_asc, axis1=1, axis2=2)
        else:
            sig = np.array(self.sigma, dtype=float)
            sig_sigT = sig @ sig.T
            trace_vals = np.einsum('ij,kji->k', sig_sigT, S_asc)

        cum    = cumulative_trapezoid(trace_vals, t_asc, initial=0.0)
        g_vals = cum[-1] - cum
        self.g_interp = interp1d(t_asc, g_vals,
                                 bounds_error=False,
                                 fill_value=(g_vals[-1], 0.0))
        return sol

    def get_value(self, t_batch, x_batch):
        """Returns value function v(t, x) = xᵀS(t)x + g(t) for a batch"""
        t_eval = (t_batch.detach().numpy().flatten()
                  if torch.is_tensor(t_batch) else np.array(t_batch).flatten())
        x_eval = (x_batch.numpy()
                  if torch.is_tensor(x_batch) else np.array(x_batch))
        x_eval = x_eval.reshape(-1, self.n)

        S_t  = self.S_interp(t_eval)
        quad = np.einsum('bi,bij,bj->b', x_eval, S_t, x_eval)
        g    = self.g_interp(t_eval)

        res = quad + g
        return torch.tensor(res, dtype=torch.float32).view(-1, 1)

    def get_control(self, t_batch, x_batch):
        """Returns optimal control a(t,x) = -D⁻¹ Mᵀ S(t) x"""
        if self.S_interp is None:
            raise ValueError("Must call solve_riccati first")

        t_np = (t_batch.detach().cpu().numpy().flatten()
                if torch.is_tensor(t_batch) else np.array(t_batch).flatten())
        x_np = (x_batch.detach().cpu().numpy()
                if torch.is_tensor(x_batch) else np.array(x_batch))
        x_np = x_np.reshape(-1, self.n)

        S_t  = self.S_interp(t_np)
        Sx   = np.einsum('bij,bj->bi', S_t, x_np)
        ctrl = -(Sx @ self.M) @ self.D_inv

        return torch.tensor(ctrl, dtype=torch.float32)


# -------------------------------------------------------
# Network architectures
# -------------------------------------------------------

class DGM_Layer(nn.Module):
    """
    Single DGM recurrent layer with gating mechanism.
    Matches the architecture from Sirignano & Spiliopoulos (2018).
    """
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

        self.gate_Z = nn.Sequential(nn.Linear(dim_x + dim_S, dim_S), self.activation)
        self.gate_G = nn.Sequential(nn.Linear(dim_x + dim_S, dim_S), self.activation)
        self.gate_R = nn.Sequential(nn.Linear(dim_x + dim_S, dim_S), self.activation)
        self.gate_H = nn.Sequential(nn.Linear(dim_x + dim_S, dim_S), self.activation)

    def forward(self, x, S):
        x_S = torch.cat([x, S], dim=1)
        Z   = self.gate_Z(x_S)
        G   = self.gate_G(x_S)
        R   = self.gate_R(x_S)
        H   = self.gate_H(torch.cat([x, S * R], dim=1))
        return (1 - G) * H + Z * S


class Net_DGM_proper(nn.Module):
    """
    Proper DGM architecture with skip connections.
    Used for value function approximation in PIA (Exercise 4).
    Matches reference [1]: Sabate-Vidales, Deep-PDE-Solvers, 2021.
    """
    def __init__(self, dim_x=2, dim_S=100, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            act = nn.Tanh()
        elif activation == 'ReLU':
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

        self.input_layer  = nn.Sequential(nn.Linear(dim_x + 1, dim_S), act)
        self.DGM1         = DGM_Layer(dim_x + 1, dim_S, activation)
        self.DGM2         = DGM_Layer(dim_x + 1, dim_S, activation)
        self.DGM3         = DGM_Layer(dim_x + 1, dim_S, activation)
        self.output_layer = nn.Linear(dim_S, 1)

    def forward(self, t, x):
        tx = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx, S1)
        S3 = self.DGM2(tx, S2)
        S4 = self.DGM3(tx, S3)
        return self.output_layer(S4)


class Net_DGM(nn.Module):
    """One hidden layer NN for Value Function approximation"""

    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t, x):
        inp = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        return self.net(inp)


class FFN(nn.Module):
    """Two hidden layers NN for Markov Control approximation"""

    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t, x):
        inp = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        return self.net(inp)


class DGMNet(nn.Module):
    """DGM Architecture for solving the PDE"""

    def __init__(self, input_dim=3, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, x):
        inp = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        return self.net(inp)


# -------------------------------------------------------
# Monte Carlo simulations
# -------------------------------------------------------

def run_monte_carlo_explicit(lqr, x_start, N_steps, M_samples, alpha_func=None):
    """Explicit Euler-Maruyama Monte Carlo simulation"""
    dt        = lqr.T / N_steps
    time_grid = np.linspace(0, lqr.T, N_steps + 1)
    lqr.solve_riccati(time_grid)

    x_start = np.atleast_1d(x_start).flatten()
    X       = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0).repeat(M_samples, 1)

    H_t   = torch.tensor(lqr.H,  dtype=torch.float32)
    M_mat = torch.tensor(lqr.M,  dtype=torch.float32)
    C_t   = torch.tensor(lqr.C,  dtype=torch.float32)
    D_t   = torch.tensor(lqr.D,  dtype=torch.float32)
    R_t   = torch.tensor(lqr.R,  dtype=torch.float32)
    sigma = (torch.eye(lqr.n) * lqr.sigma if np.isscalar(lqr.sigma)
             else torch.tensor(lqr.sigma, dtype=torch.float32))

    if not callable(alpha_func) and alpha_func is not None:
        u_const = (torch.tensor(alpha_func, dtype=torch.float32)
                   .unsqueeze(0).expand(M_samples, -1))

    total_cost = torch.zeros(M_samples)
    sqrt_dt    = np.sqrt(dt)

    for n in range(N_steps):
        t_n = time_grid[n]

        if alpha_func is None:
            t_batch = torch.full((M_samples,), t_n)
            u = lqr.get_control(t_batch, X)
        elif callable(alpha_func):
            t_batch = torch.full((M_samples,), t_n)
            u = alpha_func(t_batch, X)
        else:
            u = u_const

        cost_x      = torch.sum(X * (X @ C_t.T), dim=1)
        cost_u      = torch.sum(u * (u @ D_t.T), dim=1)
        total_cost += (cost_x + cost_u) * dt

        dW = torch.randn(M_samples, lqr.n) * sqrt_dt
        X  = X + (X @ H_t.T + u @ M_mat.T) * dt + dW @ sigma.T

    total_cost += torch.sum(X * (X @ R_t.T), dim=1)
    return torch.mean(total_cost).item(), torch.std(total_cost).item() / np.sqrt(M_samples)


def run_monte_carlo_implicit(lqr, x_start, N_steps, M_samples, alpha_func=None):
    """Implicit Euler Monte Carlo simulation"""
    dt        = lqr.T / N_steps
    time_grid = np.linspace(0, lqr.T, N_steps + 1)
    lqr.solve_riccati(time_grid)

    x_start = np.atleast_1d(x_start).flatten()
    X       = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0).repeat(M_samples, 1)

    H_t   = torch.tensor(lqr.H,     dtype=torch.float32)
    M_mat = torch.tensor(lqr.M,     dtype=torch.float32)
    C_t   = torch.tensor(lqr.C,     dtype=torch.float32)
    D_t   = torch.tensor(lqr.D,     dtype=torch.float32)
    D_inv = torch.tensor(lqr.D_inv, dtype=torch.float32)
    R_t   = torch.tensor(lqr.R,     dtype=torch.float32)
    I_n   = torch.eye(lqr.n)
    sigma = (torch.eye(lqr.n) * lqr.sigma if np.isscalar(lqr.sigma)
             else torch.tensor(lqr.sigma, dtype=torch.float32))

    if not callable(alpha_func) and alpha_func is not None:
        u_const = (torch.tensor(alpha_func, dtype=torch.float32)
                   .unsqueeze(0).expand(M_samples, -1))

    S_all = lqr.S_interp(time_grid[1:])

    total_cost = torch.zeros(M_samples)
    sqrt_dt    = np.sqrt(dt)

    for n in range(N_steps):
        t_n = time_grid[n]

        if alpha_func is None:
            t_batch = torch.full((M_samples,), t_n)
            u = lqr.get_control(t_batch, X)
        elif callable(alpha_func):
            t_batch = torch.full((M_samples,), t_n)
            u = alpha_func(t_batch, X)
        else:
            u = u_const

        cost_x      = torch.sum(X * (X @ C_t.T), dim=1)
        cost_u      = torch.sum(u * (u @ D_t.T), dim=1)
        total_cost += (cost_x + cost_u) * dt

        S_np1    = torch.tensor(S_all[n], dtype=torch.float32)
        feedback = M_mat @ D_inv @ M_mat.T @ S_np1
        A        = I_n - dt * (H_t - feedback)

        dW  = torch.randn(M_samples, lqr.n) * sqrt_dt
        rhs = X + dW @ sigma.T
        X   = torch.linalg.solve(
            A.unsqueeze(0).expand(M_samples, -1, -1),
            rhs.unsqueeze(2)
        ).squeeze(2)

    total_cost += torch.sum(X * (X @ R_t.T), dim=1)
    return torch.mean(total_cost).item(), torch.std(total_cost).item() / np.sqrt(M_samples)


# -------------------------------------------------------
# Supervised learning
# -------------------------------------------------------

def train_nn(model, t_data, x_data, target_data, epochs=1000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history   = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(t_data, x_data), target_data)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history


# -------------------------------------------------------
# DGM for linear PDE (Exercise 3)
# -------------------------------------------------------

def train_dgm_linear_pde(lqr, model, alpha, epochs=3000, batch_size=512, lr=1e-3):
    """Train DGM for the linear PDE with constant control α"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    H_t   = torch.tensor(lqr.H, dtype=torch.float32)
    M_mat = torch.tensor(lqr.M, dtype=torch.float32)
    C_t   = torch.tensor(lqr.C, dtype=torch.float32)
    D_t   = torch.tensor(lqr.D, dtype=torch.float32)
    R_t   = torch.tensor(lqr.R, dtype=torch.float32)

    if np.isscalar(lqr.sigma):
        sigma_sigmaT = torch.eye(2) * (lqr.sigma ** 2)
    else:
        s = np.array(lqr.sigma)
        sigma_sigmaT = torch.tensor(s @ s.T, dtype=torch.float32)

    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    Ma     = M_mat @ alpha_tensor
    cost_a = alpha_tensor @ D_t @ alpha_tensor

    x_test = np.array([1.0, 1.0])
    print("Computing MC reference value...")
    v_mc_ref, std_err = run_monte_carlo_explicit(
        lqr, x_test, N_steps=2000, M_samples=50000, alpha_func=alpha)
    print(f"MC reference at x={x_test}: v = {v_mc_ref:.6f} ± {std_err:.6f}")

    loss_history  = []
    error_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        t = (torch.rand(batch_size) * lqr.T * 0.99).requires_grad_(True)
        x = (torch.rand(batch_size, 2) * 6 - 3).requires_grad_(True)

        u    = model(t, x)
        ones = torch.ones_like(u)
        grads = torch.autograd.grad(u, [t, x], grad_outputs=ones, create_graph=True)
        u_t  = grads[0]
        u_x  = grads[1]

        # Correct double loop Hessian trace
        u_xx_trace = torch.zeros(batch_size)
        for i in range(2):
            grad_u_xi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0]
            for j in range(2):
                u_xx_trace += sigma_sigmaT[i, j] * grad_u_xi[:, j]

        drift    = torch.sum(u_x * (x @ H_t.T + Ma), dim=1)
        cost_x   = torch.sum(x * (x @ C_t.T), dim=1)
        residual = u_t + 0.5 * u_xx_trace + drift + cost_x + cost_a
        loss_pde = torch.mean(residual ** 2)

        t_bc   = torch.full((batch_size,), lqr.T)
        x_bc   = torch.rand(batch_size, 2) * 6 - 3
        u_bc   = model(t_bc, x_bc)
        tgt_bc = torch.sum(x_bc * (x_bc @ R_t.T), dim=1, keepdim=True)
        loss_bc = torch.mean((u_bc - tgt_bc) ** 2)

        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if epoch % 200 == 0:
            with torch.no_grad():
                t_test_t = torch.tensor([0.0])
                x_test_t = torch.tensor([x_test], dtype=torch.float32)
                v_pred   = model(t_test_t, x_test_t).item()
                rel_err  = abs(v_pred - v_mc_ref) / abs(v_mc_ref)
                error_history.append(rel_err)
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4e} | "
                  f"PDE: {loss_pde.item():.4e} | BC: {loss_bc.item():.4e} | "
                  f"v_pred: {v_pred:.4f} | RelErr: {rel_err:.4f}")

    return loss_history, error_history


# -------------------------------------------------------
# Policy Iteration with DGM (Exercise 4)
# -------------------------------------------------------

def run_policy_iteration(lqr, n_iterations=10, pde_epochs=1000, ham_epochs=500,
                         batch_size=512, lr=1e-3):
    """
    Policy Iteration Algorithm (Exercise 4.1)
    Uses proper DGM architecture with skip connections for value network.
    1. Given control a(t,x;θ_act), solve PDE for v(t,x;θ_val)
    2. Given v, update θ_act by minimising the Hamiltonian
    """
    # Use proper DGM architecture for value network
    v_net = Net_DGM_proper(dim_x=2, dim_S=128, activation='Tanh')
    a_net = FFN(input_dim=3, hidden_dim=128, output_dim=2)

    H_t = torch.tensor(lqr.H, dtype=torch.float32)
    M_t = torch.tensor(lqr.M, dtype=torch.float32)
    C_t = torch.tensor(lqr.C, dtype=torch.float32)
    D_t = torch.tensor(lqr.D, dtype=torch.float32)
    R_t = torch.tensor(lqr.R, dtype=torch.float32)

    if np.isscalar(lqr.sigma):
        sigma_sigmaT = torch.eye(2) * (lqr.sigma ** 2)
    else:
        s = np.array(lqr.sigma)
        sigma_sigmaT = torch.tensor(s @ s.T, dtype=torch.float32)

    lqr.solve_riccati(np.linspace(0, lqr.T, 1000))
    x_test = np.array([1.0, 1.0])
    v_true = lqr.get_value(torch.tensor([0.0]), torch.tensor([x_test])).item()
    a_true = lqr.get_control(torch.tensor([0.0]), torch.tensor([x_test])).numpy()

    print(f"True value at x={x_test}: v = {v_true:.6f}")
    print(f"True control at x={x_test}: a = {a_true}")

    history     = {"v_error": [], "a_error": [], "v_loss": [], "h_loss": []}
    v_optimizer = optim.Adam(v_net.parameters(), lr=lr)

    for iteration in range(n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")

        # ── Step 1: Solve PDE for value function ──────────────────────────────
        for epoch in range(pde_epochs):
            v_optimizer.zero_grad()

            t = (torch.rand(batch_size) * lqr.T * 0.99).requires_grad_(True)
            x = (torch.rand(batch_size, 2) * 6 - 3).requires_grad_(True)

            with torch.no_grad():
                a = a_net(t, x)

            v    = v_net(t, x)
            ones = torch.ones_like(v)
            grads = torch.autograd.grad(v, [t, x], grad_outputs=ones, create_graph=True)
            v_t  = grads[0]
            v_x  = grads[1]

            # Correct double loop Hessian trace
            v_xx = torch.zeros(batch_size)
            for i in range(2):
                grad_v_xi = torch.autograd.grad(v_x[:, i].sum(), x, create_graph=True)[0]
                for j in range(2):
                    v_xx += sigma_sigmaT[i, j] * grad_v_xi[:, j]

            drift    = torch.sum(v_x * (x @ H_t.T + a @ M_t.T), dim=1)
            cost_x   = torch.sum(x * (x @ C_t.T), dim=1)
            cost_a   = torch.sum(a * (a @ D_t.T), dim=1)
            residual = v_t + 0.5 * v_xx + drift + cost_x + cost_a
            loss_pde = torch.mean(residual ** 2)

            t_b      = torch.full((batch_size,), lqr.T)
            x_b      = torch.rand(batch_size, 2) * 6 - 3
            v_b      = v_net(t_b, x_b)
            target_b = torch.sum(x_b * (x_b @ R_t.T), dim=1, keepdim=True)
            loss_bc  = torch.mean((v_b - target_b) ** 2)

            loss = loss_pde + loss_bc
            loss.backward()
            v_optimizer.step()

        history["v_loss"].append(loss.item())

        # ── Step 2: Minimise Hamiltonian ──────────────────────────────────────
        a_optimizer = optim.Adam(a_net.parameters(), lr=lr)

        for epoch in range(ham_epochs):
            a_optimizer.zero_grad()

            t = torch.rand(batch_size) * lqr.T
            x = (torch.rand(batch_size, 2) * 6 - 3).requires_grad_(True)

            v   = v_net(t, x)
            v_x = torch.autograd.grad(v.sum(), x, create_graph=False)[0].detach()

            a   = a_net(t, x)
            Ma  = a @ M_t.T

            hamiltonian = torch.mean(
                torch.sum(v_x * Ma, dim=1) + torch.sum(a * (a @ D_t.T), dim=1)
            )
            hamiltonian.backward()
            a_optimizer.step()

        with torch.no_grad():
            Hx_term          = torch.sum(v_x * (x @ H_t.T), dim=1)
            cost_x_term      = torch.sum(x * (x @ C_t.T), dim=1)
            full_hamiltonian = torch.mean(
                Hx_term + torch.sum(v_x * Ma, dim=1)
                + cost_x_term + torch.sum(a * (a @ D_t.T), dim=1)
            )
        history["h_loss"].append(full_hamiltonian.item())

        # ── Evaluate errors ───────────────────────────────────────────────────
        with torch.no_grad():
            t_eval = torch.tensor([0.0])
            x_eval = torch.tensor([x_test], dtype=torch.float32)
            v_pred = v_net(t_eval, x_eval).item()
            a_pred = a_net(t_eval, x_eval).numpy().flatten()

            v_error = abs(v_pred - v_true) / abs(v_true)
            a_error = np.linalg.norm(a_pred - a_true) / np.linalg.norm(a_true)

            history["v_error"].append(v_error)
            history["a_error"].append(a_error)

            print(f"Value:   pred={v_pred:.4f}, true={v_true:.4f}, error={v_error:.4f}")
            print(f"Control: pred={a_pred}, true={a_true}, error={a_error:.4f}")

    return v_net, a_net, history


def run_policy_iteration_2(lqr, n_iterations=10, pde_epochs=3000, ham_epochs=1000,
                           batch_size=512, lr_v=5e-4, lr_a=1e-3):
    """
    Improved Policy Iteration with proper DGM architecture, persistent
    optimizer, LR scheduler, gradient clipping and early stopping.
    """
    v_net = Net_DGM_proper(dim_x=2, dim_S=128, activation='Tanh')
    a_net = FFN(input_dim=3, hidden_dim=128, output_dim=2)

    H_t = torch.tensor(lqr.H, dtype=torch.float32)
    M_t = torch.tensor(lqr.M, dtype=torch.float32)
    C_t = torch.tensor(lqr.C, dtype=torch.float32)
    D_t = torch.tensor(lqr.D, dtype=torch.float32)
    R_t = torch.tensor(lqr.R, dtype=torch.float32)

    if np.isscalar(lqr.sigma):
        sigma_sigmaT = torch.eye(2) * (lqr.sigma ** 2)
    else:
        s = np.array(lqr.sigma)
        sigma_sigmaT = torch.tensor(s @ s.T, dtype=torch.float32)

    lqr.solve_riccati(np.linspace(0, lqr.T, 1000))
    x_test = np.array([1.0, 1.0])
    v_true = lqr.get_value(torch.tensor([0.0]),
                           torch.tensor([x_test])).item()
    a_true = lqr.get_control(torch.tensor([0.0]),
                              torch.tensor([x_test])).numpy().flatten()
    print(f"True value  at x={x_test}: v = {v_true:.6f}")
    print(f"True control at x={x_test}: a = {a_true}")

    history = {
        "v_error":      [],
        "a_error":      [],
        "v_loss_final": [],
        "h_loss_final": [],
        "v_loss_curve": [],
        "h_loss_curve": [],
    }

    v_optimizer = optim.Adam(v_net.parameters(), lr=lr_v)

    for iteration in range(n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")

        for pg in v_optimizer.param_groups:
            pg['lr'] = lr_v
        v_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            v_optimizer, T_max=pde_epochs, eta_min=lr_v * 0.01)

        a_optimizer = optim.Adam(a_net.parameters(), lr=lr_a)
        a_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            a_optimizer, T_max=ham_epochs, eta_min=lr_a * 0.01)

        # ── Step 1: Solve PDE for value function ──────────────────────────────
        v_loss_this_iter = []
        best_v_loss      = float('inf')
        patience, patience_counter = 200, 0

        for epoch in range(pde_epochs):
            v_optimizer.zero_grad()

            t = (torch.rand(batch_size) * lqr.T * 0.99).requires_grad_(True)
            x = (torch.rand(batch_size, 2) * 6 - 3).requires_grad_(True)

            with torch.no_grad():
                a = a_net(t, x)

            v    = v_net(t, x)
            ones = torch.ones_like(v)
            grads = torch.autograd.grad(v, [t, x], grad_outputs=ones,
                                        create_graph=True)
            v_t  = grads[0]
            v_x  = grads[1]

            # Correct double loop Hessian trace
            v_xx = torch.zeros(batch_size)
            for i in range(2):
                grad_v_xi = torch.autograd.grad(v_x[:, i].sum(), x, create_graph=True)[0]
                for j in range(2):
                    v_xx += sigma_sigmaT[i, j] * grad_v_xi[:, j]

            drift    = torch.sum(v_x * (x @ H_t.T + a @ M_t.T), dim=1)
            cost_x   = torch.sum(x * (x @ C_t.T), dim=1)
            cost_a   = torch.sum(a * (a @ D_t.T), dim=1)
            residual = v_t + 0.5 * v_xx + drift + cost_x + cost_a
            loss_pde = torch.mean(residual ** 2)

            t_b      = torch.full((batch_size,), lqr.T)
            x_b      = torch.rand(batch_size, 2) * 6 - 3
            v_b      = v_net(t_b, x_b)
            target_b = torch.sum(x_b * (x_b @ R_t.T), dim=1, keepdim=True)
            loss_bc  = torch.mean((v_b - target_b) ** 2)

            loss = loss_pde + loss_bc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(v_net.parameters(), max_norm=1.0)
            v_optimizer.step()
            v_scheduler.step()
            v_loss_this_iter.append(loss.item())

            if loss.item() < best_v_loss - 1e-5:
                best_v_loss      = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"  PDE early stop at epoch {epoch+1}, "
                      f"loss={loss.item():.4e}")
                break

        history["v_loss_curve"].append(v_loss_this_iter)
        history["v_loss_final"].append(v_loss_this_iter[-1])
        print(f"  PDE final loss: {v_loss_this_iter[-1]:.4e}")

        # ── Step 2: Minimise Hamiltonian ──────────────────────────────────────
        h_loss_this_iter = []

        for epoch in range(ham_epochs):
            a_optimizer.zero_grad()

            t = torch.rand(batch_size) * lqr.T
            x = (torch.rand(batch_size, 2) * 6 - 3).requires_grad_(True)

            v   = v_net(t, x)
            v_x = torch.autograd.grad(v.sum(), x, create_graph=False)[0].detach()
            a   = a_net(t, x)
            Ma  = a @ M_t.T

            hamiltonian = torch.mean(
                torch.sum(v_x * Ma, dim=1) +
                torch.sum(a * (a @ D_t.T), dim=1)
            )
            hamiltonian.backward()
            torch.nn.utils.clip_grad_norm_(a_net.parameters(), max_norm=1.0)
            a_optimizer.step()
            a_scheduler.step()

            with torch.no_grad():
                full_H = torch.mean(
                    torch.sum(v_x * (x @ H_t.T), dim=1) +
                    torch.sum(v_x * Ma, dim=1) +
                    torch.sum(x * (x @ C_t.T), dim=1) +
                    torch.sum(a * (a @ D_t.T), dim=1)
                )
            h_loss_this_iter.append(full_H.item())

        history["h_loss_curve"].append(h_loss_this_iter)
        history["h_loss_final"].append(h_loss_this_iter[-1])
        print(f"  Ham final value: {h_loss_this_iter[-1]:.4e}")

        # ── Evaluate errors ───────────────────────────────────────────────────
        with torch.no_grad():
            t_eval = torch.tensor([0.0])
            x_eval = torch.tensor([x_test], dtype=torch.float32)
            v_pred = v_net(t_eval, x_eval).item()
            a_pred = a_net(t_eval, x_eval).numpy().flatten()

        v_error = abs(v_pred - v_true) / abs(v_true)
        a_error = np.linalg.norm(a_pred - a_true) / np.linalg.norm(a_true)
        history["v_error"].append(v_error)
        history["a_error"].append(a_error)

        print(f"  Value:   pred={v_pred:.4f}, true={v_true:.4f}, "
              f"rel_err={v_error:.4f}")
        print(f"  Control: pred={a_pred}, true={a_true}, "
              f"rel_err={a_error:.4f}")

    return v_net, a_net, history