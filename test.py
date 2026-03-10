import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------------------------
# LQR Class - Fixed and Optimized
# -------------------------------------------------------
class LQR:
    """Linear Quadratic Regulator Solver using Riccati ODE"""
    
    def __init__(self, H, M, C, D, R, T, sigma=0.5):
        self.H = np.array(H, dtype=np.float64)
        self.M = np.array(M, dtype=np.float64)
        self.C = np.array(C, dtype=np.float64)
        self.D = np.array(D, dtype=np.float64)
        self.R = np.array(R, dtype=np.float64)
        self.T = float(T)
        
        # Handle sigma - can be scalar or matrix
        if np.isscalar(sigma):
            self.sigma = np.eye(2) * sigma
        else:
            self.sigma = np.array(sigma, dtype=np.float64)
        
        self.sigma_sigma_T = self.sigma @ self.sigma.T
        
        self.n = self.H.shape[0]  # state dimension (2)
        self.m = self.M.shape[1]  # control dimension (2)
        self.D_inv = np.linalg.inv(self.D)

        # Validation
        self._validate_dimensions()
        self.S_interp = None
    
    def _validate_dimensions(self):
        assert self.H.shape == (self.n, self.n), f"H must be ({self.n}, {self.n})"
        assert self.M.shape[0] == self.n, "M rows must match state dimension"
        assert self.C.shape == (self.n, self.n), f"C must be ({self.n}, {self.n})"
        assert self.D.shape == (self.m, self.m), f"D must be ({self.m}, {self.m})"
        assert self.R.shape == (self.n, self.n), f"R must be ({self.n}, {self.n})"
        assert self.T > 0, "T must be positive"
        # Check positive definiteness
        assert np.all(np.linalg.eigvalsh(self.D) > 0), "D must be positive definite"
    
    def riccati_RHS(self, t, S_flat):
        """
        Riccati ODE (note: we integrate backwards from T to 0)
        S'(r) = -2 H^T S(r) + S(r) M D^{-1} M^T S(r) - C
        """
        S = S_flat.reshape(self.n, self.n)
        dS = -2 * self.H.T @ S + S @ self.M @ self.D_inv @ self.M.T @ S - self.C
        return dS.flatten()
    
    def solve_riccati(self, time_grid, rtol=1e-10, atol=1e-12):
        """Solves Riccati ODE with high precision"""
        if torch.is_tensor(time_grid):
            t_eval = time_grid.detach().cpu().numpy().flatten()
        else:
            t_eval = np.array(time_grid).flatten()
        
        t_min = np.min(t_eval)
        
        # Integrate backwards from T to t_min
        t_span = (self.T, t_min)
        
        # Dense output for better interpolation
        sol = solve_ivp(
            self.riccati_RHS,
            t_span,
            self.R.flatten(),
            method='RK45',
            rtol=rtol,
            atol=atol,
            dense_output=True
        )
        
        # Create interpolator using dense output
        def S_interp_func(t):
            t = np.atleast_1d(t)
            result = []
            for ti in t:
                ti = np.clip(ti, t_min, self.T)
                S_flat = sol.sol(ti)
                result.append(S_flat.reshape(self.n, self.n))
            return np.array(result)
        
        self.S_interp = S_interp_func
        return sol

    def get_value(self, t_batch, x_batch):
        """
        Value function: v(t,x) = x^T S(t) x + ∫_t^T tr(σσ^T S(r)) dr
        
        Args:
            t_batch: (batch_size,) or (batch_size, 1)
            x_batch: (batch_size, 1, 2) or (batch_size, 2)
        Returns:
            (batch_size, 1) tensor
        """
        if self.S_interp is None:
            raise ValueError("Must call solve_riccati first")
        
        t_np = t_batch.detach().cpu().numpy().flatten()
        x_np = x_batch.detach().cpu().numpy().reshape(-1, self.n)
        
        batch_size = len(t_np)
        S_t = self.S_interp(t_np)  # (batch, n, n)
        
        # Quadratic term: x^T S(t) x
        quad = np.einsum('bi,bij,bj->b', x_np, S_t, x_np)
        
        # Integral term: ∫_t^T tr(σσ^T S(r)) dr
        integral_term = np.zeros(batch_size)
        n_quad_points = 500  # Increase for better accuracy
        
        for i, t in enumerate(t_np):
            if t >= self.T - 1e-10:
                integral_term[i] = 0.0
            else:
                r_grid = np.linspace(t, self.T, n_quad_points)
                S_r = self.S_interp(r_grid)
                traces = np.array([np.trace(self.sigma_sigma_T @ S_r[j]) for j in range(len(r_grid))])
                integral_term[i] = trapezoid(traces, r_grid)
        
        result = quad + integral_term
        return torch.tensor(result, dtype=torch.float32).view(-1, 1)

    def get_control(self, t_batch, x_batch):
        """
        Optimal control: a(t,x) = -D^{-1} M^T S(t) x
        
        Args:
            t_batch: (batch_size,) or (batch_size, 1)
            x_batch: (batch_size, 1, 2) or (batch_size, 2)
        Returns:
            (batch_size, 2) tensor
        """
        if self.S_interp is None:
            raise ValueError("Must call solve_riccati first")
        
        t_np = t_batch.detach().cpu().numpy().flatten()
        x_np = x_batch.detach().cpu().numpy().reshape(-1, self.n)
        
        S_t = self.S_interp(t_np)  # (batch, n, n)
        
        # K(t) = -D^{-1} M^T S(t), then a = K @ x
        K = -self.D_inv @ self.M.T  # (m, n)
        control = np.einsum('ij,bjk,bk->bi', K, S_t, x_np)
        
        return torch.tensor(control, dtype=torch.float32)


# -------------------------------------------------------
# Neural Network Classes
# -------------------------------------------------------
class Net_DGM(nn.Module):
    """Network for Value Function approximation (DGM-style)"""
    
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, t, x):
        """
        Args:
            t: (batch_size,) or (batch_size, 1)
            x: (batch_size, 2) or (batch_size, 1, 2)
        Returns:
            (batch_size, 1)
        """
        t = t.view(-1, 1)
        x = x.view(x.shape[0], -1)
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)


class FFN(nn.Module):
    """Feed-forward network for Markov Control approximation"""
    
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, t, x):
        t = t.view(-1, 1)
        x = x.view(x.shape[0], -1)
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)


class DGMNet(nn.Module):
    """Deeper DGM Architecture for PDE solving"""
    
    def __init__(self, input_dim=3, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, t, x):
        t = t.view(-1, 1)
        x = x.view(x.shape[0], -1)
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)


# -------------------------------------------------------
# Monte Carlo Simulation - Fixed
# -------------------------------------------------------
def run_monte_carlo_explicit(lqr, x_start, N_steps, M_samples, alpha_func=None):
    """
    Explicit Euler-Maruyama Monte Carlo simulation.
    
    Args:
        lqr: LQR instance
        x_start: initial state (2,) or (1, 2)
        N_steps: number of time steps
        M_samples: number of MC samples
        alpha_func: None for optimal control, or callable(t, x) -> control,
                    or numpy array for constant control
    Returns:
        mean cost estimate
    """
    dt = lqr.T / N_steps
    time_grid = np.linspace(0, lqr.T, N_steps + 1)
    lqr.solve_riccati(time_grid)
    
    # Initialize state: (M_samples, 2)
    x_start = np.atleast_1d(x_start).flatten()
    X = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0).repeat(M_samples, 1)
    
    # Convert matrices to torch
    H = torch.tensor(lqr.H, dtype=torch.float32)
    M = torch.tensor(lqr.M, dtype=torch.float32)
    C = torch.tensor(lqr.C, dtype=torch.float32)
    D = torch.tensor(lqr.D, dtype=torch.float32)
    R = torch.tensor(lqr.R, dtype=torch.float32)
    sigma = torch.tensor(lqr.sigma, dtype=torch.float32)
    
    total_cost = torch.zeros(M_samples)
    
    for n in range(N_steps):
        t_n = time_grid[n]
        
        # Compute control
        if alpha_func is None:
            # Optimal control
            t_batch = torch.full((M_samples,), t_n)
            u = lqr.get_control(t_batch, X)  # (M_samples, 2)
        elif callable(alpha_func):
            t_batch = torch.full((M_samples,), t_n)
            u = alpha_func(t_batch, X)
        else:
            # Constant control
            u = torch.tensor(alpha_func, dtype=torch.float32).unsqueeze(0).repeat(M_samples, 1)
        
        # Running cost: x^T C x + u^T D u
        cost_x = torch.sum(X * (X @ C.T), dim=1)
        cost_u = torch.sum(u * (u @ D.T), dim=1)
        total_cost += (cost_x + cost_u) * dt
        
        # State update (explicit Euler)
        drift = X @ H.T + u @ M.T
        dW = torch.randn(M_samples, 2) * np.sqrt(dt)
        diffusion = dW @ sigma.T
        
        X = X + drift * dt + diffusion
    
    # Terminal cost: x_T^T R x_T
    terminal_cost = torch.sum(X * (X @ R.T), dim=1)
    total_cost += terminal_cost
    
    return torch.mean(total_cost).item(), torch.std(total_cost).item() / np.sqrt(M_samples)


def run_monte_carlo_implicit(lqr, x_start, N_steps, M_samples, alpha_func=None):
    """
    Implicit Euler Monte Carlo simulation.
    
    For the optimal control case with a = -D^{-1} M^T S(t) x:
    X_{n+1} = X_n + dt * [H X_{n+1} - M D^{-1} M^T S(t_{n+1}) X_{n+1}] + σ ΔW
    
    Rearranging: [I - dt * (H - M D^{-1} M^T S(t_{n+1}))] X_{n+1} = X_n + σ ΔW
    """
    dt = lqr.T / N_steps
    time_grid = np.linspace(0, lqr.T, N_steps + 1)
    lqr.solve_riccati(time_grid)
    
    x_start = np.atleast_1d(x_start).flatten()
    X = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0).repeat(M_samples, 1)
    
    H = torch.tensor(lqr.H, dtype=torch.float32)
    M_mat = torch.tensor(lqr.M, dtype=torch.float32)
    C = torch.tensor(lqr.C, dtype=torch.float32)
    D = torch.tensor(lqr.D, dtype=torch.float32)
    D_inv = torch.tensor(lqr.D_inv, dtype=torch.float32)
    R = torch.tensor(lqr.R, dtype=torch.float32)
    sigma = torch.tensor(lqr.sigma, dtype=torch.float32)
    I = torch.eye(lqr.n)
    
    total_cost = torch.zeros(M_samples)
    
    for n in range(N_steps):
        t_n = time_grid[n]
        t_np1 = time_grid[n + 1]
        
        # For running cost, use current state and control
        if alpha_func is None:
            t_batch = torch.full((M_samples,), t_n)
            u = lqr.get_control(t_batch, X)
        elif callable(alpha_func):
            t_batch = torch.full((M_samples,), t_n)
            u = alpha_func(t_batch, X)
        else:
            u = torch.tensor(alpha_func, dtype=torch.float32).unsqueeze(0).repeat(M_samples, 1)
        
        cost_x = torch.sum(X * (X @ C.T), dim=1)
        cost_u = torch.sum(u * (u @ D.T), dim=1)
        total_cost += (cost_x + cost_u) * dt
        
        # Implicit update
        S_np1 = torch.tensor(lqr.S_interp([t_np1])[0], dtype=torch.float32)
        feedback = M_mat @ D_inv @ M_mat.T @ S_np1
        A = I - dt * (H - feedback)
        
        dW = torch.randn(M_samples, 2) * np.sqrt(dt)
        rhs = X + dW @ sigma.T
        
        # Solve A @ X_new = rhs for each sample
        X = torch.linalg.solve(A.unsqueeze(0).expand(M_samples, -1, -1), 
                               rhs.unsqueeze(2)).squeeze(2)
    
    terminal_cost = torch.sum(X * (X @ R.T), dim=1)
    total_cost += terminal_cost
    
    return torch.mean(total_cost).item(), torch.std(total_cost).item() / np.sqrt(M_samples)


# -------------------------------------------------------
# Supervised Learning
# -------------------------------------------------------
def train_value_network(lqr, model, n_samples=10000, epochs=2000, lr=1e-3, batch_size=512):
    """Train network to approximate value function (Exercise 2.1)"""
    
    # Generate training data
    lqr.solve_riccati(np.linspace(0, lqr.T, 1000))
    
    t_data = torch.rand(n_samples) * lqr.T
    x_data = torch.rand(n_samples, 2) * 6 - 3  # Uniform on [-3, 3]^2
    
    v_target = lqr.get_value(t_data, x_data)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    history = []
    
    for epoch in range(epochs):
        # Mini-batch training
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            t_batch = t_data[idx]
            x_batch = x_data[idx]
            v_batch = v_target[idx]
            
            optimizer.zero_grad()
            v_pred = model(t_batch, x_batch)
            loss = torch.mean((v_pred - v_batch)**2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6e}")
    
    return history


def train_control_network(lqr, model, n_samples=10000, epochs=2000, lr=1e-3, batch_size=512):
    """Train network to approximate optimal control (Exercise 2.2)"""
    
    lqr.solve_riccati(np.linspace(0, lqr.T, 1000))
    
    t_data = torch.rand(n_samples) * lqr.T
    x_data = torch.rand(n_samples, 2) * 6 - 3
    
    a_target = lqr.get_control(t_data, x_data)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    history = []
    
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            
            optimizer.zero_grad()
            a_pred = model(t_data[idx], x_data[idx])
            loss = torch.mean((a_pred - a_target[idx])**2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6e}")
    
    return history


# -------------------------------------------------------
# Deep Galerkin Method - Fixed
# -------------------------------------------------------
def compute_pde_residual(model, t, x, lqr, alpha):
    """
    Compute PDE residual for the linear PDE:
    ∂_t u + (1/2) tr(σσ^T ∂_{xx} u) + (∂_x u)^T H x + (∂_x u)^T M α + x^T C x + α^T D α = 0
    
    Args:
        model: neural network u(t, x; θ)
        t: (batch,) tensor, requires_grad=True
        x: (batch, 2) tensor, requires_grad=True
        lqr: LQR instance
        alpha: (2,) constant control
    """
    batch_size = t.shape[0]
    
    # Convert matrices
    H = torch.tensor(lqr.H, dtype=torch.float32)
    M = torch.tensor(lqr.M, dtype=torch.float32)
    C = torch.tensor(lqr.C, dtype=torch.float32)
    D = torch.tensor(lqr.D, dtype=torch.float32)
    sigma_sigma_T = torch.tensor(lqr.sigma_sigma_T, dtype=torch.float32)
    
    alpha = torch.tensor(alpha, dtype=torch.float32)
    
    # Forward pass
    u = model(t, x)  # (batch, 1)
    
    # First derivatives
    grad_outputs = torch.ones_like(u)
    grads = torch.autograd.grad(u, [t, x], grad_outputs=grad_outputs, create_graph=True)
    u_t = grads[0]  # (batch,)
    u_x = grads[1]  # (batch, 2)
    
    # Second derivatives (Hessian diagonal for Laplacian)
    # We need tr(σσ^T ∂_{xx} u) = Σ_{i,j} (σσ^T)_{ij} ∂²u/∂x_i∂x_j
    # For diagonal σσ^T, this simplifies to Σ_i (σσ^T)_{ii} ∂²u/∂x_i²
    
    u_xx = torch.zeros(batch_size)
    for i in range(2):
        grad_u_xi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0]
        for j in range(2):
            u_xx += sigma_sigma_T[i, j] * grad_u_xi[:, j]
    
    # Drift term: (∂_x u)^T H x
    Hx = x @ H.T  # (batch, 2)
    drift_x = torch.sum(u_x * Hx, dim=1)
    
    # Control term: (∂_x u)^T M α
    Ma = M @ alpha  # (2,)
    drift_a = torch.sum(u_x * Ma, dim=1)
    
    # Running costs
    cost_x = torch.sum(x * (x @ C.T), dim=1)  # x^T C x
    cost_a = alpha @ D @ alpha  # α^T D α (scalar)
    
    # PDE residual
    residual = u_t + 0.5 * u_xx + drift_x + drift_a + cost_x + cost_a
    
    return residual


def train_dgm_linear_pde(lqr, model, alpha, epochs=3000, batch_size=512, lr=1e-3):
    """
    Train DGM for the linear PDE (Exercise 3.1)
    
    Args:
        lqr: LQR instance
        model: neural network
        alpha: constant control (2,)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    R = torch.tensor(lqr.R, dtype=torch.float32)
    
    loss_history = []
    error_history = []
    
    # Reference solution via MC
    x_test = np.array([1.0, 1.0])
    v_mc_ref, _ = run_monte_carlo_explicit(lqr, x_test, N_steps=1000, M_samples=50000,
                                           alpha_func=np.array(alpha))
    print(f"MC reference value at x={x_test}: {v_mc_ref:.6f}")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Sample interior points
        t = torch.rand(batch_size, requires_grad=True) * lqr.T * 0.99  # Avoid t=T
        x = (torch.rand(batch_size, 2) * 6 - 3).requires_grad_(True)
        
        # PDE residual loss
        residual = compute_pde_residual(model, t, x, lqr, alpha)
        loss_pde = torch.mean(residual**2)
        
        # Boundary condition loss at t=T
        t_boundary = torch.full((batch_size,), lqr.T)
        x_boundary = torch.rand(batch_size, 2) * 6 - 3
        u_boundary = model(t_boundary, x_boundary)
        terminal_target = torch.sum(x_boundary * (x_boundary @ R.T), dim=1, keepdim=True)
        loss_boundary = torch.mean((u_boundary - terminal_target)**2)
        
        loss = loss_pde + loss_boundary
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        
        if epoch % 200 == 0:
            # Compute error vs MC reference
            with torch.no_grad():
                t_test = torch.tensor([0.0])
                x_test_t = torch.tensor([x_test], dtype=torch.float32)
                v_pred = model(t_test, x_test_t).item()
                rel_error = abs(v_pred - v_mc_ref) / abs(v_mc_ref)
                error_history.append(rel_error)
            
            print(f"Epoch {epoch}: Loss={loss.item():.4e}, PDE={loss_pde.item():.4e}, "
                  f"BC={loss_boundary.item():.4e}, RelErr={rel_error:.4f}")
    
    return loss_history, error_history


# -------------------------------------------------------
# Policy Iteration with DGM - Fixed
# -------------------------------------------------------
def run_policy_iteration(lqr, n_iterations=10, pde_epochs=1000, ham_epochs=500,
                         batch_size=512, lr=1e-3):
    """
    Policy Iteration Algorithm (Exercise 4.1)
    
    1. Given control a(t,x;θ_act), solve PDE for v(t,x;θ_val)
    2. Given v, update θ_act by minimizing the Hamiltonian
    """
    v_net = Net_DGM(input_dim=3, hidden_dim=128, output_dim=1)
    a_net = FFN(input_dim=3, hidden_dim=128, output_dim=2)
    
    # Convert matrices
    H = torch.tensor(lqr.H, dtype=torch.float32)
    M = torch.tensor(lqr.M, dtype=torch.float32)
    C = torch.tensor(lqr.C, dtype=torch.float32)
    D = torch.tensor(lqr.D, dtype=torch.float32)
    R = torch.tensor(lqr.R, dtype=torch.float32)
    sigma_sigma_T = torch.tensor(lqr.sigma_sigma_T, dtype=torch.float32)
    
    # Reference solution
    lqr.solve_riccati(np.linspace(0, lqr.T, 1000))
    x_test = np.array([1.0, 1.0])
    v_true = lqr.get_value(torch.tensor([0.0]), torch.tensor([x_test])).item()
    a_true = lqr.get_control(torch.tensor([0.0]), torch.tensor([x_test])).numpy()
    
    print(f"True value at x={x_test}: v = {v_true:.6f}")
    print(f"True control at x={x_test}: a = {a_true}")
    
    history = {"v_error": [], "a_error": [], "v_loss": [], "h_loss": []}
    
    for iteration in range(n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")
        
        # Step 1: Solve PDE for value function given current control
        v_optimizer = optim.Adam(v_net.parameters(), lr=lr)
        
        for epoch in range(pde_epochs):
            v_optimizer.zero_grad()
            
            # Sample points
            t = torch.rand(batch_size, requires_grad=True) * lqr.T * 0.99
            x = (torch.rand(batch_size, 2) * 6 - 3).requires_grad_(True)
            
            # Get current control (detached)
            with torch.no_grad():
                a = a_net(t, x)  # (batch, 2)
            
            # Forward pass
            v = v_net(t, x)
            
            # Gradients
            grad_outputs = torch.ones_like(v)
            grads = torch.autograd.grad(v, [t, x], grad_outputs=grad_outputs, create_graph=True)
            v_t = grads[0]
            v_x = grads[1]
            
            # Hessian trace
            v_xx = torch.zeros(batch_size)
            for i in range(2):
                grad_v_xi = torch.autograd.grad(v_x[:, i].sum(), x, create_graph=True)[0]
                for j in range(2):
                    v_xx += sigma_sigma_T[i, j] * grad_v_xi[:, j]
            
            # Drift terms
            Hx = x @ H.T
            Ma = a @ M.T
            drift = torch.sum(v_x * (Hx + Ma), dim=1)
            
            # Running cost
            cost_x = torch.sum(x * (x @ C.T), dim=1)
            cost_a = torch.sum(a * (a @ D.T), dim=1)
            
            # PDE residual
            residual = v_t + 0.5 * v_xx + drift + cost_x + cost_a
            loss_pde = torch.mean(residual**2)
            
            # Boundary condition
            t_b = torch.full((batch_size,), lqr.T)
            x_b = torch.rand(batch_size, 2) * 6 - 3
            v_b = v_net(t_b, x_b)
            target_b = torch.sum(x_b * (x_b @ R.T), dim=1, keepdim=True)
            loss_bc = torch.mean((v_b - target_b)**2)
            
            loss = loss_pde + loss_bc
            loss.backward()
            v_optimizer.step()
        
        history["v_loss"].append(loss.item())
        
        # Step 2: Update control by minimizing Hamiltonian
        a_optimizer = optim.Adam(a_net.parameters(), lr=lr)
        
        for epoch in range(ham_epochs):
            a_optimizer.zero_grad()
            
            t = torch.rand(batch_size) * lqr.T
            x = (torch.rand(batch_size, 2) * 6 - 3).requires_grad_(True)
            
            # Get gradient of value function (detached)
            v = v_net(t, x)
            v_x = torch.autograd.grad(v.sum(), x, create_graph=False)[0].detach()
            
            # Current control
            a = a_net(t, x)
            
            # Hamiltonian (only terms involving a)
            # H = v_x^T M a + a^T D a
            Ma = a @ M.T
            hamiltonian = torch.mean(torch.sum(v_x * Ma, dim=1) + torch.sum(a * (a @ D.T), dim=1))
            
            hamiltonian.backward()
            a_optimizer.step()
        
        history["h_loss"].append(hamiltonian.item())
        
        # Evaluate errors
        with torch.no_grad():
            t_eval = torch.tensor([0.0])
            x_eval = torch.tensor([x_test], dtype=torch.float32)
            
            v_pred = v_net(t_eval, x_eval).item()
            a_pred = a_net(t_eval, x_eval).numpy().flatten()
            
            v_error = abs(v_pred - v_true) / abs(v_true)
            a_error = np.linalg.norm(a_pred - a_true) / np.linalg.norm(a_true)
            
            history["v_error"].append(v_error)
            history["a_error"].append(a_error)
            
            print(f"Value: pred={v_pred:.4f}, true={v_true:.4f}, error={v_error:.4f}")
            print(f"Control: pred={a_pred}, true={a_true}, error={a_error:.4f}")
    
    return v_net, a_net, history


# -------------------------------------------------------
# Plotting Utilities
# -------------------------------------------------------
def plot_mc_convergence_time_steps(lqr, x_start, M_samples=100000):
    """Exercise 1.2: Convergence vs number of time steps"""
    lqr.solve_riccati(np.linspace(0, lqr.T, 10000))
    v_true = lqr.get_value(torch.tensor([0.0]), torch.tensor([x_start])).item()
    
    N_steps_list = [1, 10, 50, 100, 500, 1000, 5000]
    errors_explicit = []
    errors_implicit = []
    
    print(f"True value: {v_true:.6f}")
    print("Testing time step convergence...")
    
    for N in N_steps_list:
        v_mc_exp, _ = run_monte_carlo_explicit(lqr, x_start, N, M_samples)
        v_mc_imp, _ = run_monte_carlo_implicit(lqr, x_start, N, M_samples)
        
        err_exp = abs(v_mc_exp - v_true)
        err_imp = abs(v_mc_imp - v_true)
        
        errors_explicit.append(err_exp)
        errors_implicit.append(err_imp)
        
        print(f"N={N}: Explicit={v_mc_exp:.4f} (err={err_exp:.4e}), "
              f"Implicit={v_mc_imp:.4f} (err={err_imp:.4e})")
    
    plt.figure(figsize=(10, 6))
    plt.loglog(N_steps_list, errors_explicit, 'bo-', label='Explicit Euler', linewidth=2)
    plt.loglog(N_steps_list, errors_implicit, 'rs-', label='Implicit Euler', linewidth=2)
    
    # Reference line for O(dt) = O(1/N)
    ref_line = np.array(N_steps_list, dtype=float)**(-1) * errors_explicit[2] * N_steps_list[2]
    plt.loglog(N_steps_list, ref_line, 'k--', label='O(1/N) reference', alpha=0.7)
    
    plt.xlabel('Number of Time Steps (N)', fontsize=12)
    plt.ylabel('Absolute Error |V_MC - V_true|', fontsize=12)
    plt.title('Monte Carlo Convergence: Time Step Refinement\n(Fixed M = {:,} samples)'.format(M_samples), fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig('mc_convergence_time_steps.png', dpi=150)
    plt.show()
    
    return N_steps_list, errors_explicit, errors_implicit


def plot_mc_convergence_samples(lqr, x_start, N_steps=5000):
    """Exercise 1.2: Convergence vs number of MC samples"""
    lqr.solve_riccati(np.linspace(0, lqr.T, 10000))
    v_true = lqr.get_value(torch.tensor([0.0]), torch.tensor([x_start])).item()
    
    M_samples_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    errors = []
    std_errors = []
    
    print(f"True value: {v_true:.6f}")
    print("Testing MC sample convergence...")
    
    for M in M_samples_list:
        v_mc, std_mc = run_monte_carlo_explicit(lqr, x_start, N_steps, M)
        err = abs(v_mc - v_true)
        errors.append(err)
        std_errors.append(std_mc)
        print(f"M={M}: V_MC={v_mc:.4f}, err={err:.4e}, std_err={std_mc:.4e}")
    
    plt.figure(figsize=(10, 6))
    plt.loglog(M_samples_list, errors, 'bo-', label='Absolute Error', linewidth=2)
    plt.loglog(M_samples_list, std_errors, 'g^-', label='Standard Error (σ/√M)', linewidth=2)
    
    # Reference line for O(1/sqrt(M))
    ref_line = np.array(M_samples_list, dtype=float)**(-0.5) * std_errors[4] * np.sqrt(M_samples_list[4])
    plt.loglog(M_samples_list, ref_line, 'k--', label='O(1/√M) reference', alpha=0.7)
    
    plt.xlabel('Number of Monte Carlo Samples (M)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Monte Carlo Convergence: Sample Size\n(Fixed N = {} time steps)'.format(N_steps), fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig('mc_convergence_samples.png', dpi=150)
    plt.show()
    
    return M_samples_list, errors, std_errors


# -------------------------------------------------------
# Main Execution
# -------------------------------------------------------
if __name__ == "__main__":
    # Define LQR problem parameters
    H = [[0.1, 0.], [0., 0.1]]
    M = np.eye(2)
    C = np.eye(2)
    D = np.eye(2) * 0.1
    R = np.eye(2)
    T = 1.0
    sigma = 0.5
    
    # Create LQR instance
    lqr = LQR(H, M, C, D, R, T, sigma)
    
    # Test point
    x_start = np.array([1.0, 1.0])
    
    print("="*60)
    print("EXERCISE 1.1: Riccati ODE Solution")
    print("="*60)
    
    time_grid = np.linspace(0, T, 100)
    lqr.solve_riccati(time_grid)
    
    t_test = torch.tensor([0.0, 0.5, 1.0])
    x_test_batch = torch.tensor([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    
    v_values = lqr.get_value(t_test, x_test_batch)
    a_values = lqr.get_control(t_test, x_test_batch)
    
    print("Value function at test points:")
    for i in range(len(t_test)):
        print(f"  v({t_test[i].item():.1f}, {x_test_batch[i].numpy()}) = {v_values[i].item():.6f}")
    
    print("\nOptimal control at test points:")
    for i in range(len(t_test)):
        print(f"  a({t_test[i].item():.1f}, {x_test_batch[i].numpy()}) = {a_values[i].numpy()}")
    
    print("\n" + "="*60)
    print("EXERCISE 1.2: Monte Carlo Convergence")
    print("="*60)
    
    # Quick test
    v_true = lqr.get_value(torch.tensor([0.0]), torch.tensor([x_start])).item()
    v_mc, std_mc = run_monte_carlo_explicit(lqr, x_start, N_steps=1000, M_samples=10000)
    print(f"True value: {v_true:.6f}")
    print(f"MC estimate: {v_mc:.6f} ± {std_mc:.6f}")
    print(f"Relative error: {abs(v_mc - v_true)/v_true:.4f}")
    
    # Uncomment to run full convergence studies (takes time)
    # plot_mc_convergence_time_steps(lqr, x_start, M_samples=100000)
    # plot_mc_convergence_samples(lqr, x_start, N_steps=5000)
    
    print("\n" + "="*60)
    print("EXERCISE 2.1: Supervised Learning - Value Function")
    print("="*60)
    
    v_model = Net_DGM(input_dim=3, hidden_dim=100, output_dim=1)
    v_loss_history = train_value_network(lqr, v_model, n_samples=10000, epochs=2000)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(v_loss_history)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Supervised Learning: Value Function Approximation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('supervised_value.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("EXERCISE 2.2: Supervised Learning - Control")
    print("="*60)
    
    a_model = FFN(input_dim=3, hidden_dim=100, output_dim=2)
    a_loss_history = train_control_network(lqr, a_model, n_samples=10000, epochs=2000)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(a_loss_history)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Supervised Learning: Control Function Approximation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('supervised_control.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("EXERCISE 3.1: Deep Galerkin Method - Linear PDE")
    print("="*60)
    
    alpha_const = np.array([1.0, 1.0])
    dgm_model = DGMNet(input_dim=3, hidden_dim=100)
    dgm_loss, dgm_errors = train_dgm_linear_pde(lqr, dgm_model, alpha_const, epochs=3000)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.semilogy(dgm_loss)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('DGM Training Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(0, len(dgm_errors)*200, 200), dgm_errors, 'ro-')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Relative Error vs MC', fontsize=12)
    ax2.set_title('DGM Error vs Monte Carlo Reference', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dgm_linear_pde.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("EXERCISE 4.1: Policy Iteration with DGM")
    print("="*60)
    
    v_net, a_net, pia_history = run_policy_iteration(
        lqr, n_iterations=10, pde_epochs=1000, ham_epochs=500
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].semilogy(range(1, len(pia_history["v_error"])+1), pia_history["v_error"], 'bo-')
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Relative Value Error', fontsize=12)
    axes[0, 0].set_title('Value Function Convergence', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].semilogy(range(1, len(pia_history["a_error"])+1), pia_history["a_error"], 'rs-')
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Relative Control Error', fontsize=12)
    axes[0, 1].set_title('Control Convergence', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].semilogy(range(1, len(pia_history["v_loss"])+1), pia_history["v_loss"], 'g^-')
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('PDE Loss', fontsize=12)
    axes[1, 0].set_title('Value PDE Training Loss', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(range(1, len(pia_history["h_loss"])+1), pia_history["h_loss"], 'm*-')
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Hamiltonian', fontsize=12)
    axes[1, 1]