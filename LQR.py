import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------
# Classes
# -------------------------------------------------------
class LQR:
    """Exercise 1.1: Linear Quadratic Regulator Solver using Riccati ODE"""
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
        """Solves Riccati ODE on a specified time grid"""
        t_eval = time_grid.detach().cpu().numpy() if torch.is_tensor(time_grid) else np.array(time_grid)
        t_span = (self.T, np.min(t_eval))
        sol = solve_ivp(self.ricat_RHS, t_span, self.R.flatten(), t_eval=np.sort(t_eval)[::-1])
        S_values = sol.y.T.reshape(-1, self.n, self.n)
        self.S_interp = interp1d(sol.t, S_values, axis=0, bounds_error=False, fill_value="extrapolate")
        return sol

    def get_value(self, t_batch, x_batch):
        """Returns value function v(t, x) for a batch"""
        S_t = self.S_interp(t_batch.detach().cpu().numpy())
        res = np.einsum('bik,bkj,bij->b', x_batch.cpu().numpy(), S_t, x_batch.cpu().numpy())
        return torch.tensor(res, dtype=torch.float32).view(-1, 1)

    def get_control(self, t_batch, x_batch):
        """Returns optimal Markov control a(t, x) for a batch"""
        S_t = self.S_interp(t_batch.detach().cpu().numpy())
        K = -self.D_inv @ self.M.T @ S_t
        ctrl = np.einsum('bij,bkj->bi', K, x_batch.cpu().numpy())
        return torch.tensor(ctrl, dtype=torch.float32)

class Net_DGM(nn.Module):
    """One hidden layer NN for Value Function approximation (Exercise 2.1 & 4.1)"""
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super(Net_DGM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, t, x):
        inp = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        return self.net(inp)

class FFN(nn.Module):
    """Two hidden layers NN for Markov Control approximation (Exercise 2.2 & 4.1)"""
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
    """DGM Architecture for solving the PDE (Exercise 3.1)"""
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

def run_monte_carlo_explicit(lqr, x_start, N_steps, M_samples, alpha_val=None):
    """ Vectorized MC Simulator to calculate M_samples paths.
    """
    dt = lqr.T / N_steps
    # Pre-solve Riccati and move to torch once [cite: 44, 64]
    time_grid = torch.linspace(0, lqr.T, N_steps + 1)
    lqr.solve_riccati(time_grid)
    
    # Pre-interpolate S(t) for all time steps to avoid overhead in the loop
    S_t_all = torch.tensor(lqr.S_interp(time_grid.numpy()), dtype=torch.float32)
    
    # Initialize state: (M_samples, 1, 2) [cite: 45]
    x = torch.tensor(x_start, dtype=torch.float32).repeat(M_samples, 1, 1)
    total_cost = torch.zeros(M_samples)
    
    # Constants as tensors
    H = torch.tensor(lqr.H).float()
    M = torch.tensor(lqr.M).float()
    C = torch.tensor(lqr.C).float()
    D = torch.tensor(lqr.D).float()
    D_inv = torch.tensor(lqr.D_inv).float()
    R = torch.tensor(lqr.R).float()
    sigma = lqr.sigma

    for i in range(N_steps):
        S_curr = S_t_all[i]
        
        # Determine Control alpha
        if alpha_val is None:
            # Optimal control: a = -D^-1 * M.T * S(t) * x
            # K shape: (2, 2). x shape: (M, 1, 2)
            K = -D_inv @ M.T @ S_curr
            u = torch.einsum('ij, bkj -> bi', K, x)
        else:
            u = torch.tensor(alpha_val).float().repeat(M_samples, 1)
        
        # Accumulate running cost
        # X^T * C * X
        term_x = torch.bmm(x, C @ x.transpose(1, 2)).view(-1)
        # alpha^T * D * alpha
        term_u = torch.bmm(u.unsqueeze(1), D @ u.unsqueeze(2)).view(-1)
        total_cost += (term_x + term_u) * dt
        
        # Update State with Explicit Euler
        # dX = [HX + Ma]dt + sigma*dW
        drift = (torch.einsum('ij, bkj -> bi', H, x) + torch.einsum('ij, bj -> bi', M, u)) * dt
        diffusion = sigma * np.sqrt(dt) * torch.randn(M_samples, 2)
        x = x + (drift + diffusion).unsqueeze(1)

    # Add Terminal Cost: X_T^T * R * X_T [cite: 33, 39]
    total_cost += torch.bmm(x, R @ x.transpose(1, 2)).view(-1)
    
    return torch.mean(total_cost).item()

def run_monte_carlo_implicit(lqr, x_start, N_steps, M_samples, alpha_val=None):
    """
    Vectorized Implicit MC Simulator that solves the linear system
    for the implicit step for all samples at once.
    """
    dt = lqr.T / N_steps
    time_grid = torch.linspace(0, lqr.T, N_steps + 1)
    lqr.solve_riccati(time_grid)
    
    # Pre-interpolate S(t) and move to torch
    S_t_all = torch.tensor(lqr.S_interp(time_grid.numpy()), dtype=torch.float32)
    
    x = torch.tensor(x_start, dtype=torch.float32).repeat(M_samples, 1, 1)
    total_cost = torch.zeros(M_samples)
    
    # Constants
    H = torch.tensor(lqr.H).float()
    M = torch.tensor(lqr.M).float()
    C = torch.tensor(lqr.C).float()
    D = torch.tensor(lqr.D).float()
    D_inv = torch.tensor(lqr.D_inv).float()
    R = torch.tensor(lqr.R).float()
    I = torch.eye(lqr.n)
    sigma = lqr.sigma

    for i in range(N_steps):
        # We need S at t_{n+1} for the implicit step 
        S_next = S_t_all[i+1]
        S_curr = S_t_all[i]
        
        # Calculate Control
        if alpha_val is None:
            K_curr = -D_inv @ M.T @ S_curr
            u = torch.einsum('ij, bkj -> bi', K_curr, x)
        else:
            u = torch.tensor(alpha_val).float().repeat(M_samples, 1)

        # Accumulate running cost
        term_x = torch.bmm(x, C @ x.transpose(1, 2)).view(-1)
        term_u = torch.bmm(u.unsqueeze(1), D @ u.unsqueeze(2)).view(-1)
        total_cost += (term_x + term_u) * dt

        # Implicit Update Step
        # Calculate the matrix: A = [I - dt(H - M D^-1 M^T S_next)]
        # For constant alpha, the term is [I - dt*H], but the prompt implies feedback form.
        feedback_gain = M @ D_inv @ M.T @ S_next
        A = I - dt * (H - feedback_gain)
        
        # Right hand side: X_n + sigma * dW
        dW = sigma * np.sqrt(dt) * torch.randn(M_samples, 2)
        rhs = (x.squeeze(1) + dW).unsqueeze(2) # Shape: (M, 2, 1)
        
        # Solve A * X_{n+1} = rhs
        # torch.linalg.solve handles the batch of systems efficiently
        x_next = torch.linalg.solve(A, rhs)
        x = x_next.transpose(1, 2) # Back to (M, 1, 2)

    # Terminal Cost [cite: 33, 39]
    total_cost += torch.bmm(x, R @ x.transpose(1, 2)).view(-1)
    
    return torch.mean(total_cost).item()

def train_nn(model, t_data, x_data, target_data, epochs=1000, lr=1e-3):
    """Supervised Learning trainer"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(t_data, x_data), target_data)
        loss.backward(); optimizer.step()
        history.append(loss.item())
    return history

def train_dgm_linear_pde(lqr_obj, dgm_model, x_test_point, alpha_val, v_mc_target, epochs=1001, batch_size=512):
    optimizer = torch.optim.Adam(dgm_model.parameters(), lr=1e-3)
    loss_history = []
    error_history = []
    
    # Constants to device
    H = torch.tensor(lqr_obj.H, device=device).float()
    M = torch.tensor(lqr_obj.M, device=device).float()
    C = torch.tensor(lqr_obj.C, device=device).float()
    D = torch.tensor(lqr_obj.D, device=device).float()
    R = torch.tensor(lqr_obj.R, device=device).float()
    alpha = torch.tensor(alpha_val, device=device).float().view(-1, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Interior Loss (PDE Residual)
        t = torch.rand(batch_size, device=device, requires_grad=True) * lqr_obj.T
        x = (torch.rand(batch_size, 1, 2, device=device) * 6 - 3).requires_grad_(True)
        
        u = dgm_model(t, x)
        grads = torch.autograd.grad(u.sum(), [t, x], create_graph=True)
        u_t = grads[0].view(-1, 1)
        u_x = grads[1].view(-1, 2, 1)
        
        # 2D Laplacian
        u_xx_1 = torch.autograd.grad(u_x[:, 0].sum(), x, create_graph=True)[0][:, 0, 0]
        u_xx_2 = torch.autograd.grad(u_x[:, 1].sum(), x, create_graph=True)[0][:, 0, 1]
        lap = (u_xx_1 + u_xx_2).view(-1, 1)
        
        # Drift and Costs
        drift = torch.bmm(u_x.transpose(1, 2), (H @ x.transpose(1, 2) + M @ alpha)).view(-1, 1)
        running = torch.bmm(x, C @ x.transpose(1, 2)).view(-1, 1) + (alpha.t() @ D @ alpha)
        
        loss_pde = torch.mean((u_t + 0.5 * (lqr_obj.sigma**2) * lap + drift + running)**2)
        
        # 2. Boundary Loss (Terminal Condition)
        t_b = torch.full((batch_size,), lqr_obj.T, device=device)
        x_b = torch.rand(batch_size, 1, 2, device=device) * 6 - 3
        loss_b = torch.mean((dgm_model(t_b, x_b) - torch.bmm(x_b, R @ x_b.transpose(1, 2)).view(-1, 1))**2)
        
        loss = loss_pde + loss_b
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # 3. Validation Error (Every 200 epochs)
        if epoch % 200 == 0:
            t_0 = torch.tensor([0.0], device=device)
            x_0 = torch.tensor([x_test_point], device=device).float()
            v_pred = dgm_model(t_0, x_0).item()
            rel_error = abs(v_pred - v_mc_target) / abs(v_mc_target)
            error_history.append(rel_error)
            print(f"Epoch {epoch} | Loss: {loss.item():.4e} | Rel Error: {rel_error:.4f}")
            
    return loss_history, error_history

def run_policy_iteration(lqr_obj, x_test_point, pia_iterations=5, pde_epochs=500, ham_epochs=200):
    """
    Implements Exercise 4.1: Policy Iteration using DGM for LQR.
    """
    # Initialize networks as per assignment specs [cite: 82, 88]
    v_net = Net_DGM(hidden_dim=100) # [cite: 82]
    a_net = FFN(hidden_dim=100)     # [cite: 88]
    
    v_opt = optim.Adam(v_net.parameters(), lr=1e-3)
    a_opt = optim.Adam(a_net.parameters(), lr=1e-3)
    
    # Constants from LQR object [cite: 31, 33, 35]
    H = torch.tensor(lqr_obj.H).float().to(device)
    M = torch.tensor(lqr_obj.M).float().to(device)
    C = torch.tensor(lqr_obj.C).float().to(device)
    D = torch.tensor(lqr_obj.D).float().to(device)
    R_term = torch.tensor(lqr_obj.R).float().to(device)
    sigma = lqr_obj.sigma

    # True values for convergence check [cite: 37, 41]
    lqr_obj.solve_riccati(torch.linspace(0, lqr_obj.T, 100))
    v_true = lqr_obj.get_value(torch.tensor([0.0]), torch.tensor([x_test_point])).item()
    a_true = lqr_obj.get_control(torch.tensor([0.0]), torch.tensor([x_test_point])).numpy()

    pia_history = {"v_err": [], "a_err": []}

    for i in range(pia_iterations):
        # Policy Evaluation
        for _ in range(pde_epochs):
            v_opt.zero_grad()
            t = torch.rand(512, device=device, requires_grad=True) * lqr_obj.T
            x = torch.rand(512, 1, 2, device=device, requires_grad=True) * 6 - 3
            
            v_val = v_net(t, x)
            a_val = a_net(t, x).view(-1, 2, 1).detach()
            
            # Gradients
            grad_v = torch.autograd.grad(v_val.sum(), [t, x], create_graph=True)
            v_t, v_x = grad_v[0].view(-1, 1), grad_v[1].view(-1, 2, 1)
            
            # 2D Laplacian
            v_xx_1 = torch.autograd.grad(v_x[:, 0].sum(), x, create_graph=True)[0][:, 0, 0]
            v_xx_2 = torch.autograd.grad(v_x[:, 1].sum(), x, create_graph=True)[0][:, 0, 1]
            lap = (v_xx_1 + v_xx_2).view(-1, 1)
            
            # Linearized PDE Residual
            drift = torch.bmm(v_x.transpose(1, 2), (H @ x.transpose(1, 2) + M @ a_val)).view(-1, 1)
            running = torch.bmm(x, C @ x.transpose(1, 2)).view(-1, 1) + \
                      torch.bmm(a_val.transpose(1, 2), D @ a_val).view(-1, 1)
            
            loss_pde = torch.mean((v_t + 0.5 * (sigma**2) * lap + drift + running)**2)
            
            # Boundary Condition
            x_b = (torch.rand(256, 1, 2, device=device) * 6 - 3)
            t_b = torch.full((256,), lqr_obj.T, device=device)
            loss_b = torch.mean((v_net(torch.full((256,), lqr_obj.T), x_b) - \
                                torch.bmm(x_b, R_term @ x_b.transpose(1, 2)).view(-1, 1))**2)
            
            (loss_pde + loss_b).backward()
            v_opt.step()

        # Policy Improvement step
        for _ in range(ham_epochs):
            a_opt.zero_grad()
            t_ham = torch.rand(512, device=device) * lqr_obj.T
            x_ham = torch.rand(512, 1, 2, device=device, requires_grad=True) * 6 - 3
            
            # Fix Value Function gradients
            v_val_fix = v_net(t_ham, x_ham)
            v_x_fix = torch.autograd.grad(v_val_fix.sum(), x_ham)[0].view(-1, 2, 1).detach()
            
            # Hamiltonian Minimization
            a_curr = a_net(t_ham, x_ham).view(-1, 2, 1)
            hamiltonian = torch.mean(
                torch.bmm(v_x_fix.transpose(1, 2), M @ a_curr).view(-1, 1) + \
                torch.bmm(a_curr.transpose(1, 2), D @ a_curr).view(-1, 1)
            )
            
            hamiltonian.backward()
            a_opt.step()

        # Convergence Check
        with torch.no_grad():
            v_curr = v_net(torch.tensor([0.0], device=device), 
               torch.tensor([x_test_point], device=device)).item()
            a_curr = a_net(torch.tensor([0.0], device=device), torch.tensor([x_test_point], device=device)).cpu().numpy()
            
            v_err = abs(v_curr - v_true) / abs(v_true)
            a_err = np.linalg.norm(a_curr - a_true) / np.linalg.norm(a_true)
            
            pia_history["v_err"].append(v_err)
            pia_history["a_err"].append(a_err)
            print(f"Iteration {i+1}: Value Error = {v_err:.4f}, Action Error = {a_err:.4f}")

    return pia_history
