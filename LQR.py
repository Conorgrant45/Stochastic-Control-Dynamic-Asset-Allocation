import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Classes
# -------------------------------------------------------
class LQR:
    """Linear Quadratic Regulator Solver using Riccati ODE"""
    def __init__(self, H, M, C, D, R, T, sigma=0.5):
        self.H, self.M = np.array(H, dtype=float), np.array(M, dtype=float)
        self.C, self.D, self.R = np.array(C, dtype=float), np.array(D, dtype=float), np.array(R, dtype=float)
        self.T, self.sigma = float(T), sigma
        self.n, self.m = self.H.shape[0], self.M.shape[1]
        self.D_inv = np.linalg.inv(self.D)

        # Validation steps
        assert self.M.shape[0] == self.n, "M rows must match H"
        assert self.C.shape == (self.n, self.n), "C wrong dimensions"
        assert self.D.shape == (self.m, self.m), "D wrong dimensions"
        assert self.R.shape == (self.n, self.n), "R wrong dimensions"
        assert self.T > 0, "T must be positive"
        self.S_interp = None
    
    def ricat_RHS(self, t, S_flat):
        """Right-hand side of the Riccati ODE"""

        S = S_flat.reshape(self.n, self.n)
        Sprime = -2 * self.H.T @ S + S @ self.M @ self.D_inv @ self.M.T @ S - self.C
        return Sprime.reshape(-1)
    
    def solve_riccati(self, time_grid):
        """Solves Riccati ODE on a specified time grid with high precision"""

        t_eval = time_grid.detach().numpy() if torch.is_tensor(time_grid) else np.array(time_grid)
        t_span = (self.T, np.min(t_eval))
        
        sol = solve_ivp(
            self.ricat_RHS, 
            t_span, 
            self.R.flatten(), 
            t_eval=np.sort(t_eval)[::-1],
            method='RK45',    # Explicit Runge-Kutta method
            rtol=1e-9,        # Relative tolerance
            atol=1e-12        # Absolute tolerance
        )
        
        S_values = sol.y.T.reshape(-1, self.n, self.n)
        self.S_interp = interp1d(sol.t, S_values, axis=0, bounds_error=False, fill_value="extrapolate")
        return sol

    def get_value(self, t_batch, x_batch):
        """
        Returns value function v(t, x) for a batch.
        v(t,x) = x' S(t) x + integral of tr(sigma^2 * S(r)) 
        """
        t_eval = t_batch.detach().numpy().flatten() if torch.is_tensor(t_batch) else np.array(t_batch).flatten()
        x_eval = x_batch.numpy() if torch.is_tensor(x_batch) else np.array(x_batch)
        
        # Solve Riccati and get S(t)
        S_t = self.S_interp(t_eval)
        
        # Quadratic term: x^T * S(t) * x
        quad = np.einsum('...ik,...kj,...ij->...', x_eval, S_t, x_eval)
        quad = quad.flatten() # Ensure this is (N,)
        
        # Stochastic term: Integral of tr(sigma^2 * S(r)) from t to T
        stochastic_term = []
        for t in t_eval:
            if t >= self.T:
                stochastic_term.append(0.0)
                continue
            
            # Numerical integration of the trace
            r_grid = np.linspace(t, self.T, 100)
            S_r = self.S_interp(r_grid)
            trace_S = np.trace(S_r, axis1=1, axis2=2)
            integrand = (self.sigma**2) * trace_S
            
            val = trapezoid(integrand, r_grid)
            stochastic_term.append(val)
            
        stochastic_term = np.array(stochastic_term)
        
        # Combine: (N,) + (N,) = (N,)
        res = quad + stochastic_term
        return torch.tensor(res, dtype=torch.float32).view(-1, 1)

    
    def get_control(self, t_batch, x_batch):
        """
        Returns optimal control a(t,x) = -D^{-1} M^T S(t) x
        """
        if self.S_interp is None:
            raise ValueError("Must call solve_riccati first")
    
        t_np = t_batch.detach().cpu().numpy().flatten()
        x_np = x_batch.detach().cpu().numpy().reshape(-1, self.n)  # (batch, n)
    
        S_t = self.S_interp(t_np)  # (batch, n, n)
    
        # a(t,x) = -D^{-1} M^T S(t) x
        # Step 1: S(t) @ x for each batch element
        Sx = np.einsum('bij,bj->bi', S_t, x_np)  # (batch, n)
    
        # Step 2: M^T @ (S(t) @ x) for each batch element
        MTSx = Sx @ self.M  # (batch, m)
    
        # Step 3: -D^{-1} @ (M^T @ S(t) @ x) for each batch element
        ctrl = -MTSx @ self.D_inv  # (batch, m)
    
        return torch.tensor(ctrl, dtype=torch.float32)
    
class Net_DGM(nn.Module):
    """One hidden layer NN for Value Function approximation"""

    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super(Net_DGM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, t, x):
        inp = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        return self.net(inp)

class FFN(nn.Module):
    """Two hidden layers NN for Markov Control approximation"""

    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, t, x):
        inp = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        return self.net(inp)

class DGMNet(nn.Module):
    """DGM Architecture for solving the PDE"""

    def __init__(self, input_dim=3, hidden_dim=100):
        super(DGMNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, t, x):
        inp = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        return self.net(inp)

# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
def run_monte_carlo_explicit(lqr, x_start, N_steps, M_samples, alpha_func=None):
    """
    Explicit Euler-Maruyama Monte Carlo simulation.
    
    Args:
        alpha_func: None for optimal control, callable(t, x), or numpy array for constant control
    Returns:
        (mean_cost, std_error) tuple
    """
    dt = lqr.T / N_steps
    time_grid = np.linspace(0, lqr.T, N_steps + 1)
    lqr.solve_riccati(time_grid)
    
    # Handle x_start shape
    x_start = np.atleast_1d(x_start).flatten()
    X = torch.tensor(x_start, dtype=torch.float32).unsqueeze(0).repeat(M_samples, 1)
    
    # Convert matrices
    H = torch.tensor(lqr.H, dtype=torch.float32)
    M_mat = torch.tensor(lqr.M, dtype=torch.float32)
    C = torch.tensor(lqr.C, dtype=torch.float32)
    D = torch.tensor(lqr.D, dtype=torch.float32)
    R = torch.tensor(lqr.R, dtype=torch.float32)
    
    # Handle sigma (scalar or matrix)
    if np.isscalar(lqr.sigma):
        sigma = torch.eye(2) * lqr.sigma
    else:
        sigma = torch.tensor(lqr.sigma, dtype=torch.float32)
    
    total_cost = torch.zeros(M_samples)
    
    for n in range(N_steps):
        t_n = time_grid[n]
        
        # Compute control
        if alpha_func is None:
            # Optimal control from LQR
            t_batch = torch.full((M_samples,), t_n)
            u = lqr.get_control(t_batch, X)
        elif callable(alpha_func):
            # Neural network or function control
            t_batch = torch.full((M_samples,), t_n)
            u = alpha_func(t_batch, X)
        else:
            # Constant control (numpy array)
            u = torch.tensor(alpha_func, dtype=torch.float32).unsqueeze(0).repeat(M_samples, 1)
        
        # Running cost: x^T C x + u^T D u
        cost_x = torch.sum(X * (X @ C.T), dim=1)
        cost_u = torch.sum(u * (u @ D.T), dim=1)
        total_cost += (cost_x + cost_u) * dt
        
        # State update (explicit Euler-Maruyama)
        drift = X @ H.T + u @ M_mat.T
        dW = torch.randn(M_samples, 2) * np.sqrt(dt)
        diffusion = dW @ sigma.T
        
        X = X + drift * dt + diffusion
    
    # Terminal cost
    terminal_cost = torch.sum(X * (X @ R.T), dim=1)
    total_cost += terminal_cost
    
    mean_cost = torch.mean(total_cost).item()
    std_error = torch.std(total_cost).item() / np.sqrt(M_samples)
    
    return mean_cost, std_error

def run_monte_carlo_implicit(lqr, x_start, N_steps, M_samples, alpha_func=None):
    """
    Implicit Euler Monte Carlo simulation.
    Returns: (mean_cost, std_error) tuple
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
    I = torch.eye(lqr.n)
    
    if np.isscalar(lqr.sigma):
        sigma = torch.eye(2) * lqr.sigma
    else:
        sigma = torch.tensor(lqr.sigma, dtype=torch.float32)
    
    total_cost = torch.zeros(M_samples)
    
    for n in range(N_steps):
        t_n = time_grid[n]
        t_np1 = time_grid[n + 1]
        
        # Control at current time
        if alpha_func is None:
            t_batch = torch.full((M_samples,), t_n)
            u = lqr.get_control(t_batch, X)
        elif callable(alpha_func):
            t_batch = torch.full((M_samples,), t_n)
            u = alpha_func(t_batch, X)
        else:
            u = torch.tensor(alpha_func, dtype=torch.float32).unsqueeze(0).repeat(M_samples, 1)
        
        # Running cost
        cost_x = torch.sum(X * (X @ C.T), dim=1)
        cost_u = torch.sum(u * (u @ D.T), dim=1)
        total_cost += (cost_x + cost_u) * dt
        
        # Implicit update (for optimal control case)
        S_np1 = torch.tensor(lqr.S_interp([t_np1])[0], dtype=torch.float32)
        feedback = M_mat @ D_inv @ M_mat.T @ S_np1
        A = I - dt * (H - feedback)
        
        dW = torch.randn(M_samples, 2) * np.sqrt(dt)
        rhs = X + dW @ sigma.T
        
        X = torch.linalg.solve(A.unsqueeze(0).expand(M_samples, -1, -1), 
                               rhs.unsqueeze(2)).squeeze(2)
    
    # Terminal cost
    terminal_cost = torch.sum(X * (X @ R.T), dim=1)
    total_cost += terminal_cost
    
    mean_cost = torch.mean(total_cost).item()
    std_error = torch.std(total_cost).item() / np.sqrt(M_samples)
    
    return mean_cost, std_error
def train_nn(model, t_data, x_data, target_data, epochs=1000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(t_data, x_data), target_data)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history

def train_dgm_linear_pde(lqr, model, alpha, epochs=3000, batch_size=512, lr=1e-3):
    """
    Train DGM for the linear PDE with constant control α.
    
    PDE: ∂_t u + (1/2)tr(σσ^T ∂_xx u) + (∂_x u)^T Hx + (∂_x u)^T Mα + x^T Cx + α^T Dα = 0
    BC:  u(T, x) = x^T R x
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    # Convert to torch tensors
    H = torch.tensor(lqr.H, dtype=torch.float32)
    M_mat = torch.tensor(lqr.M, dtype=torch.float32)
    C = torch.tensor(lqr.C, dtype=torch.float32)
    D = torch.tensor(lqr.D, dtype=torch.float32)
    R = torch.tensor(lqr.R, dtype=torch.float32)
    
    # Handle sigma
    if np.isscalar(lqr.sigma):
        sigma_sigma_T = torch.eye(2) * (lqr.sigma ** 2)
    else:
        sigma_np = np.array(lqr.sigma)
        sigma_sigma_T = torch.tensor(sigma_np @ sigma_np.T, dtype=torch.float32)
    
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
    
    # Compute MC reference at test point
    x_test = np.array([1.0, 1.0])
    print("Computing MC reference value...")
    v_mc_ref, std_err = run_monte_carlo_explicit(
        lqr, x_test, N_steps=2000, M_samples=50000, alpha_func=alpha
    )
    print(f"MC reference at x={x_test}: v = {v_mc_ref:.6f} ± {std_err:.6f}")
    
    loss_history = []
    error_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Sample interior points (avoid t=T)
        t = (torch.rand(batch_size) * lqr.T * 0.99).requires_grad_(True)
        x = (torch.rand(batch_size, 2) * 6 - 3).requires_grad_(True)  # Uniform on [-3, 3]^2
        
        # Forward pass
        u = model(t, x)  # (batch, 1)
        
        # Compute gradients
        grad_outputs = torch.ones_like(u)
        grads = torch.autograd.grad(u, [t, x], grad_outputs=grad_outputs, create_graph=True)
        u_t = grads[0]  # (batch,)
        u_x = grads[1]  # (batch, 2)
        
        # Compute Hessian trace: tr(σσ^T ∂_xx u)
        u_xx_trace = torch.zeros(batch_size)
        for i in range(2):
            grad_u_xi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0]
            for j in range(2):
                u_xx_trace += sigma_sigma_T[i, j] * grad_u_xi[:, j]
        
        # Drift terms
        Hx = x @ H.T  # (batch, 2)
        Ma = M_mat @ alpha_tensor  # (2,)
        drift = torch.sum(u_x * (Hx + Ma), dim=1)
        
        # Running costs
        cost_x = torch.sum(x * (x @ C.T), dim=1)  # x^T C x
        cost_a = alpha_tensor @ D @ alpha_tensor  # α^T D α (scalar)
        
        # PDE residual
        residual = u_t + 0.5 * u_xx_trace + drift + cost_x + cost_a
        loss_pde = torch.mean(residual ** 2)
        
        # Boundary condition at t = T
        t_bc = torch.full((batch_size,), lqr.T)
        x_bc = torch.rand(batch_size, 2) * 6 - 3
        u_bc = model(t_bc, x_bc)
        target_bc = torch.sum(x_bc * (x_bc @ R.T), dim=1, keepdim=True)  # x^T R x
        loss_bc = torch.mean((u_bc - target_bc) ** 2)
        
        # Total loss
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        
        # Compute error vs MC reference periodically
        if epoch % 200 == 0:
            with torch.no_grad():
                t_test = torch.tensor([0.0])
                x_test_t = torch.tensor([x_test], dtype=torch.float32)
                v_pred = model(t_test, x_test_t).item()
                rel_error = abs(v_pred - v_mc_ref) / abs(v_mc_ref)
                error_history.append(rel_error)
            
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4e} | "
                  f"PDE: {loss_pde.item():.4e} | BC: {loss_bc.item():.4e} | "
                  f"v_pred: {v_pred:.4f} | RelErr: {rel_error:.4f}")
    
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
    
    if np.isscalar(lqr.sigma):
        sigma_sigma_T = torch.eye(2) * (lqr.sigma ** 2)
    else:
        sigma_np = np.array(lqr.sigma)
        sigma_sigma_T = torch.tensor(sigma_np @ sigma_np.T, dtype=torch.float32)
    
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