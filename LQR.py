import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LQR:
    def __init__(self, H, M, C, D, R, T, sigma=0.5):
        self.H, self.M = np.array(H, dtype=float), np.array(M, dtype=float)
        self.C, self.D, self.R = np.array(C, dtype=float), np.array(D, dtype=float), np.array(R, dtype=float)
        self.T, self.sigma = float(T), sigma
        self.n, self.m = self.H.shape[0], self.M.shape[1]
        self.D_inv = np.linalg.inv(self.D)
        self.S_interp = None
    
    def ricat_RHS(self, t, S_flat):
        S = S_flat.reshape(self.n, self.n)
        Sprime = -2 * self.H.T @ S + S @ self.M @ self.D_inv @ self.M.T @ S - self.C
        return Sprime.reshape(-1)
    
    def solve_riccati(self, time_grid):
        t_eval = time_grid.detach().cpu().numpy() if torch.is_tensor(time_grid) else np.array(time_grid)
        t_span = (self.T, np.min(t_eval))
        sol = solve_ivp(self.ricat_RHS, t_span, self.R.flatten(), t_eval=np.sort(t_eval)[::-1])
        S_values = sol.y.T.reshape(-1, self.n, self.n)
        self.S_interp = interp1d(sol.t, S_values, axis=0, bounds_error=False, fill_value="extrapolate")

    def get_value(self, t_batch, x_batch):
        S_t = self.S_interp(t_batch.detach().cpu().numpy())
        res = np.einsum('bik,bkj,bij->b', x_batch.cpu().numpy(), S_t, x_batch.cpu().numpy())
        return torch.tensor(res, dtype=torch.float32).view(-1, 1)

    def get_control(self, t_batch, x_batch):
        S_t = self.S_interp(t_batch.detach().cpu().numpy())
        K = -self.D_inv @ self.M.T @ S_t
        ctrl = np.einsum('bij,bkj->bi', K, x_batch.cpu().numpy())
        return torch.tensor(ctrl, dtype=torch.float32)

class Net_DGM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))
    def forward(self, t, x):
        inp = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        return self.net(inp)

class FFN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, t, x):
        inp = torch.cat([t.view(-1, 1), x.view(x.shape[0], -1)], dim=1)
        return self.net(inp)

def run_policy_iteration(lqr_obj, x_test_point, pia_iterations=5, pde_epochs=500, ham_epochs=200):
    v_net = Net_DGM(hidden_dim=100).to(device)
    a_net = FFN(hidden_dim=100).to(device)
    v_opt, a_opt = optim.Adam(v_net.parameters(), lr=1e-3), optim.Adam(a_net.parameters(), lr=1e-3)
    
    H = torch.tensor(lqr_obj.H, device=device).float()
    M = torch.tensor(lqr_obj.M, device=device).float()
    C = torch.tensor(lqr_obj.C, device=device).float()
    D = torch.tensor(lqr_obj.D, device=device).float()
    R_term = torch.tensor(lqr_obj.R, device=device).float()
    
    lqr_obj.solve_riccati(torch.linspace(0, lqr_obj.T, 100))
    v_true = lqr_obj.get_value(torch.tensor([0.0]), torch.tensor([x_test_point])).item()
    a_true = lqr_obj.get_control(torch.tensor([0.0]), torch.tensor([x_test_point])).numpy()

    pia_history = {"v_err": [], "a_err": []}
    for i in range(pia_iterations):
        for _ in range(pde_epochs):
            v_opt.zero_grad()
            t = torch.rand(512, device=device, requires_grad=True) * lqr_obj.T
            x = torch.rand(512, 1, 2, device=device, requires_grad=True) * 6 - 3
            v_val = v_net(t, x)
            grad_v = torch.autograd.grad(v_val.sum(), [t, x], create_graph=True)
            v_t, v_x = grad_v[0].view(-1, 1), grad_v[1].view(-1, 2, 1)
            lap = (torch.autograd.grad(v_x[:, 0].sum(), x, create_graph=True)[0][:, 0, 0] + 
                   torch.autograd.grad(v_x[:, 1].sum(), x, create_graph=True)[0][:, 0, 1]).view(-1, 1)
            drift = torch.bmm(v_x.transpose(1, 2), (H @ x.transpose(1, 2) + M @ a_net(t, x).view(-1, 2, 1).detach())).view(-1, 1)
            running = torch.bmm(x, C @ x.transpose(1, 2)).view(-1, 1) + torch.bmm(a_net(t, x).view(-1, 2, 1).detach().transpose(1, 2), D @ a_net(t, x).view(-1, 2, 1).detach()).view(-1, 1)
            loss_pde = torch.mean((v_t + 0.5 * (lqr_obj.sigma**2) * lap + drift + running)**2)
            
            x_b = torch.rand(256, 1, 2, device=device) * 6 - 3
            t_b = torch.full((256,), lqr_obj.T, device=device)
            loss_b = torch.mean((v_net(t_b, x_b) - torch.bmm(x_b, R_term @ x_b.transpose(1, 2)).view(-1, 1))**2)
            (loss_pde + loss_b).backward(); v_opt.step()

        for _ in range(ham_epochs):
            a_opt.zero_grad()
            t_h = torch.rand(512, device=device) * lqr_obj.T
            x_h = torch.rand(512, 1, 2, device=device, requires_grad=True) * 6 - 3
            v_x_f = torch.autograd.grad(v_net(t_h, x_h).sum(), x_h)[0].view(-1, 2, 1).detach()
            a_c = a_net(t_h, x_h).view(-1, 2, 1)
            torch.mean(torch.bmm(v_x_f.transpose(1, 2), M @ a_c).view(-1, 1) + torch.bmm(a_c.transpose(1, 2), D @ a_c).view(-1, 1)).backward()
            a_opt.step()

        with torch.no_grad():
            v_curr = v_net(torch.tensor([0.0], device=device), torch.tensor([x_test_point], device=device)).item()
            a_curr = a_net(torch.tensor([0.0], device=device), torch.tensor([x_test_point], device=device)).cpu().numpy()
            pia_history["v_err"].append(abs(v_curr - v_true) / abs(v_true))
            pia_history["a_err"].append(np.linalg.norm(a_curr - a_true) / np.linalg.norm(a_true))
    return pia_history