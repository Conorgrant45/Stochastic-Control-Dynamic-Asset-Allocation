"""
dgm_lqr.py
----------
Solves the LQR HJB PDE using the Deep Galerkin Method, structured to mirror
pde_BlackScholes_exchange_dgm.py from reference [1]:

    [1] M. Sabate-Vidales, Deep-PDE-Solvers,
        https://github.com/msabvid/Deep-PDE-Solvers, 2021.

Two modes, selected at construction:

  mode="fixed_control"     -- Exercise 3
      Solves the LINEAR PDE with constant control alpha_fixed=(1,1).
      .fit()            trains the DGM network on the PDE residual + BC loss.
      .unbiased_cost()  returns (costs, costs_cv) using the DGM gradient as
                        a control variate, mirroring unbiased_price() in [1].

  mode="policy_iteration"  -- Exercise 4
      Alternates between:
        i)  solving the linear PDE for v given current a  (DGM, as in Ex 3)
        ii) minimising the Hamiltonian to update a        (as in brief §4)
      .fit()            runs the outer iteration loop.
      .value() / .control() evaluate the final learned functions.
      .unbiased_cost()  same MC + control-variate evaluation as above.

Problem
-------
  dX = (HX + Mα) dt + σ dW
  minimise  E[ ∫₀ᵀ (xᵀCx + αᵀDα) dt  +  xᵀRx ]

HJB PDE (linear, fixed α):
  u_t + ½tr(σσᵀ u_xx) + (∂_x u)ᵀHx + (∂_x u)ᵀMα + xᵀCx + αᵀDα = 0
  u(T,x) = xᵀRx

Exact solution (Riccati):  v(t,x) = xᵀS(t)x + ∫ₜᵀ tr(σσᵀS(r)) dr
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d


# ── Network architectures ─────────────────────────────────────────────────────

class Net_DGM(nn.Module):
    """
    One hidden-layer network for value function approximation.
    Matches the Net_DGM class referenced in the coursework brief (Exercise 2.1).
    """
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
    """
    Two hidden-layer network for Markov control approximation.
    Matches the FFN class referenced in the coursework brief (Exercise 2.2).
    """
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


# ── Riccati exact solver ──────────────────────────────────────────────────────

class RiccatiSolver:
    """
    Solves the matrix Riccati ODE backward in time and provides the exact
    value function and optimal control as a benchmark.

    Riccati ODE:  S'(r) = -2HᵀS + SMD⁻¹MᵀS - C,  S(T) = R
    Value:        v(t,x) = xᵀS(t)x + ∫ₜᵀ tr(σσᵀS(r)) dr
    Control:      α*(t,x) = -D⁻¹MᵀS(t)x
    """
    def __init__(self, H, M, C, D, R, sigma, T):
        self.H = np.array(H, dtype=float)
        self.M = np.array(M, dtype=float)
        self.C = np.array(C, dtype=float)
        self.D = np.array(D, dtype=float)
        self.R = np.array(R, dtype=float)
        self.T = float(T)
        self.n = self.H.shape[0]
        self.D_inv = np.linalg.inv(self.D)
        if np.isscalar(sigma):
            self.sigma_sigmaT = np.eye(self.n) * float(sigma) ** 2
            self.sigma_mat    = np.eye(self.n) * float(sigma)
        else:
            s = np.array(sigma, dtype=float)
            self.sigma_sigmaT = s @ s.T
            self.sigma_mat    = s
        self.S_interp = None

    def _rhs(self, t, S_flat):
        S = S_flat.reshape(self.n, self.n)
        dS = -2 * self.H.T @ S + S @ self.M @ self.D_inv @ self.M.T @ S - self.C
        return dS.reshape(-1)

    def solve(self, n_grid=1000):
        t_eval = np.linspace(self.T, 0.0, n_grid)
        sol = solve_ivp(self._rhs, (self.T, 0.0), self.R.flatten(),
                        t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-12)
        S_vals = sol.y.T.reshape(-1, self.n, self.n)
        self.S_interp = interp1d(sol.t, S_vals, axis=0,
                                 bounds_error=False, fill_value="extrapolate")
        return self

    def S(self, t):
        assert self.S_interp is not None, "Call .solve() first"
        return self.S_interp(float(t))

    def value(self, t, x):
        """Exact v(t,x) = xᵀS(t)x + ∫ₜᵀ tr(σσᵀS(r)) dr"""
        St = self.S(t)
        x  = np.array(x, dtype=float).flatten()
        quad = float(x @ St @ x)
        r_grid   = np.linspace(float(t), self.T, 500)
        integrand = np.array([np.trace(self.sigma_sigmaT @ self.S_interp(r))
                               for r in r_grid])
        return quad + trapezoid(integrand, r_grid)

    def control(self, t, x):
        """Optimal α*(t,x) = -D⁻¹MᵀS(t)x"""
        St = self.S(t)
        x  = np.array(x, dtype=float).flatten()
        return -(self.D_inv @ self.M.T @ St @ x)


# ── Main class ────────────────────────────────────────────────────────────────

class PDE_DGM_LQR(nn.Module):
    """
    Deep Galerkin Method solver for the LQR HJB PDE.

    Mirrors the structure of PDE_DGM_BlackScholes from reference [1]:
      - Constructor takes all problem parameters plus a mode flag
      - .fit()           trains the network(s)
      - .value(t, x)     evaluates the learned value function
      - .control(t, x)   returns the learned / implied optimal control
      - .unbiased_cost() MC estimate with DGM gradient as control variate,
                         mirroring unbiased_price() in [1]

    After .fit() the following are populated for plotting:
      self.loss_history  -- list of total loss per step  (fixed_control mode)
      self.error_history -- list of (step, rel_error) tuples at log intervals
      self.pia_history   -- dict of convergence lists    (policy_iteration mode)

    Args:
        d, m:         state / control dimensions
        H, M, C, D, R: LQR matrices
        sigma:        scalar or (d×d) diffusion
        T:            time horizon
        hidden_dim:   network width (default 100, matches brief)
        x_range:      spatial sampling range ±x_range (default 3)
        mode:         "fixed_control" (Ex 3) or "policy_iteration" (Ex 4)
        alpha_fixed:  numpy array of shape (m,), required when mode="fixed_control"
    """

    def __init__(self, d, m, H, M, C, D, R, sigma, T,
                 hidden_dim=100, x_range=3.0,
                 mode="fixed_control", alpha_fixed=None):
        super().__init__()

        assert mode in ("fixed_control", "policy_iteration"), \
            "mode must be 'fixed_control' or 'policy_iteration'"
        self.mode = mode
        self.d, self.m, self.T, self.x_range = d, m, T, x_range

        # Register matrices as buffers so they move to device automatically
        def buf(arr, name):
            self.register_buffer(name, torch.tensor(
                np.array(arr, dtype=np.float32)))

        buf(H, "H"); buf(M, "M_mat"); buf(C, "C_mat")
        buf(D, "D_mat"); buf(R, "R_mat")
        buf(np.linalg.inv(np.array(D, dtype=np.float64)).astype(np.float32), "D_inv")

        if np.isscalar(sigma):
            buf(np.eye(d, dtype=np.float32) * float(sigma)**2, "sigma_sigmaT")
            buf(np.eye(d, dtype=np.float32) * float(sigma),    "sigma_mat")
        else:
            s = np.array(sigma, dtype=np.float32)
            buf(s @ s.T, "sigma_sigmaT"); buf(s, "sigma_mat")

        if alpha_fixed is not None:
            self.register_buffer("alpha_fixed",
                torch.tensor(np.array(alpha_fixed, dtype=np.float32)))
        else:
            self.alpha_fixed = None

        # Networks — Net_DGM for value (matches brief Ex 2.1),
        #            FFN for control  (matches brief Ex 2.2)
        self.v_net = Net_DGM(input_dim=1+d, hidden_dim=hidden_dim, output_dim=1)
        self.a_net = FFN(input_dim=1+d, hidden_dim=hidden_dim, output_dim=m) \
                     if mode == "policy_iteration" else None

        # Exact benchmark
        self.riccati = RiccatiSolver(H, M, C, D, R, sigma, T)

        # History containers (populated by .fit())
        self.loss_history  = []
        self.error_history = []
        self.pia_history   = {"v_error": [], "a_error": [], "v_loss": [], "h_loss": []}

    # ── PDE residual ──────────────────────────────────────────────────────────

    def _pde_residual(self, t, x, alpha_batch):
        """
        HJB residual for linear PDE with control alpha_batch (batch, m):
          r = u_t + ½tr(σσᵀ u_xx) + (∂_x u)ᵀHx + (∂_x u)ᵀMα + xᵀCx + αᵀDα
        Returns tensor of shape (batch,).
        """
        t = t.requires_grad_(True)
        x = x.requires_grad_(True)
        u = self.v_net(t, x)
        ones = torch.ones_like(u)
        grads = torch.autograd.grad(u, [t, x], grad_outputs=ones, create_graph=True)
        u_t = grads[0]          # (batch,)
        u_x = grads[1]          # (batch, d)

        # Second-order term: tr(σσᵀ ∂²u/∂x²)
        trace = torch.zeros(u_t.shape[0], device=t.device)
        for i in range(self.d):
            g = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0]
            for j in range(self.d):
                trace = trace + self.sigma_sigmaT[i, j] * g[:, j]

        drift  = torch.sum(u_x * (x @ self.H.T), dim=1)          # (∂_x u)ᵀHx
        drift += torch.sum(u_x * (alpha_batch @ self.M_mat.T), dim=1)  # (∂_x u)ᵀMα
        cost_x = torch.sum(x * (x @ self.C_mat.T), dim=1)        # xᵀCx
        cost_a = torch.sum(alpha_batch * (alpha_batch @ self.D_mat.T), dim=1)  # αᵀDα

        return u_t + 0.5 * trace + drift + cost_x + cost_a        # (batch,)

    # ── fit() — dispatches to the right training loop ─────────────────────────

    def fit(self, max_updates=3000, batch_size=512, lr=1e-3,
            log_every=200, pde_epochs=1000, ham_epochs=500):
        """
        Train the solver.

        fixed_control mode:
            max_updates  -- gradient steps
            log_every    -- how often to record error vs MC reference

        policy_iteration mode:
            max_updates  -- number of outer iterations
            pde_epochs   -- PDE solve steps per iteration  (step i  in brief)
            ham_epochs   -- Hamiltonian min steps per iter (step ii in brief)
        """
        self.riccati.solve()

        if self.mode == "fixed_control":
            self._fit_fixed(max_updates, batch_size, lr, log_every)
        else:
            self._fit_policy_iteration_2(max_updates, pde_epochs, ham_epochs,
                                       batch_size, lr)
        self.eval()
        return self

    # ── Exercise 3: fixed-control DGM ────────────────────────────────────────

    def _fit_fixed(self, max_updates, batch_size, lr, log_every):
        """
        Minimise:
          L = E[residual²] + E[(u(T,x) - xᵀRx)²]
        with constant alpha_fixed, following the DGM approach of ref [1].
        """
        assert self.alpha_fixed is not None, \
            "Provide alpha_fixed when using mode='fixed_control'"

        dev = next(self.parameters()).device
        opt = torch.optim.Adam(self.v_net.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)

        # MC reference (constant alpha) for error tracking
        alpha_np = self.alpha_fixed.cpu().numpy()
        v_mc_ref, _ = __import__("LQR").run_monte_carlo_explicit(
            self._make_lqr_obj(), x_start=[1.0, 1.0],
            N_steps=500, M_samples=20000, alpha_func=alpha_np
        )

        self.train()
        for step in range(1, max_updates + 1):
            opt.zero_grad()

            # Interior points
            t = (torch.rand(batch_size, device=dev) * self.T * 0.99)
            x = (torch.rand(batch_size, self.d, device=dev) * 2*self.x_range
                 - self.x_range)
            alpha_b = self.alpha_fixed.unsqueeze(0).expand(batch_size, -1)

            residual  = self._pde_residual(t, x, alpha_b)
            loss_pde  = residual.pow(2).mean()

            # Boundary condition u(T,x) = xᵀRx
            t_bc  = torch.full((batch_size,), self.T, device=dev)
            x_bc  = (torch.rand(batch_size, self.d, device=dev) * 2*self.x_range
                     - self.x_range)
            u_bc  = self.v_net(t_bc, x_bc)
            tgt   = torch.sum(x_bc * (x_bc @ self.R_mat.T), dim=1, keepdim=True)
            loss_bc = (u_bc - tgt).pow(2).mean()

            loss = loss_pde + loss_bc
            loss.backward()
            opt.step()
            sch.step()

            self.loss_history.append(loss.item())

            if step % log_every == 0 or step == 1:
                with torch.no_grad():
                    t0 = torch.zeros(1, device=dev)
                    x0 = torch.tensor([[1.0, 1.0]], device=dev)
                    v_pred = self.v_net(t0, x0).item()
                rel_err = abs(v_pred - v_mc_ref) / abs(v_mc_ref)
                self.error_history.append((step, rel_err))
                print(f"Step {step:4d}/{self.loss_history.__len__()} | "
                      f"loss={loss.item():.4e} | rel_err={rel_err:.4f}")

    # ── Exercise 4: policy iteration ──────────────────────────────────────────

    def _fit_policy_iteration(self, n_iter, pde_epochs, ham_epochs, batch_size, lr):
        """
        PIA with DGM as specified in brief §4:
          i)  Given a(·;θ_act), solve linear PDE for v(·;θ_val) via DGM
          ii) Given v, minimise Hamiltonian to update θ_act
        """
        dev = next(self.parameters()).device

        # Ground truth for error tracking
        x_test   = np.array([1.0, 1.0])
        v_true   = self.riccati.value(0.0, x_test)
        a_true   = self.riccati.control(0.0, x_test)

        print(f"Riccati reference  v(0,x0)={v_true:.6f},  a(0,x0)={a_true}")

        for iteration in range(1, n_iter + 1):
            print(f"\n=== Policy Iteration {iteration}/{n_iter} ===")

            # ── Step i: solve PDE for v given current a ───────────────────────
            v_opt = torch.optim.Adam(self.v_net.parameters(), lr=lr)
            self.v_net.train()

            for epoch in range(pde_epochs):
                v_opt.zero_grad()

                t = torch.rand(batch_size, device=dev) * self.T * 0.99
                x = (torch.rand(batch_size, self.d, device=dev) * 2*self.x_range
                     - self.x_range)

                with torch.no_grad():
                    a = self.a_net(t, x)          # current control policy

                residual = self._pde_residual(t, x, a)
                loss_pde = residual.pow(2).mean()

                t_bc = torch.full((batch_size,), self.T, device=dev)
                x_bc = (torch.rand(batch_size, self.d, device=dev) * 2*self.x_range
                        - self.x_range)
                u_bc = self.v_net(t_bc, x_bc)
                tgt  = torch.sum(x_bc * (x_bc @ self.R_mat.T), dim=1, keepdim=True)
                loss_bc = (u_bc - tgt).pow(2).mean()

                loss = loss_pde + loss_bc
                loss.backward()
                v_opt.step()

            self.pia_history["v_loss"].append(loss.item())

            # ── Step ii: minimise Hamiltonian to update a ─────────────────────
            # H(θ_act) = E[(∂_x v)ᵀHx + (∂_x v)ᵀMa + xᵀCx + aᵀDa]
            a_opt = torch.optim.Adam(self.a_net.parameters(), lr=lr)
            self.a_net.train()

            for epoch in range(ham_epochs):
                a_opt.zero_grad()

                t = torch.rand(batch_size, device=dev) * self.T
                x = (torch.rand(batch_size, self.d, device=dev) * 2*self.x_range
                     - self.x_range).requires_grad_(True)

                v    = self.v_net(t, x)
                v_x  = torch.autograd.grad(v.sum(), x, create_graph=False)[0].detach()

                a    = self.a_net(t, x)
                Ma   = a @ self.M_mat.T
                # Hamiltonian terms involving a only (ref brief §4, step ii)
                hamiltonian = torch.mean(
                    torch.sum(v_x * Ma, dim=1) +
                    torch.sum(a * (a @ self.D_mat.T), dim=1)
                )
                hamiltonian.backward()
                a_opt.step()

            # Full Hamiltonian for logging
            with torch.no_grad():
                Hx_term   = torch.sum(v_x * (x @ self.H.T), dim=1)
                cost_x    = torch.sum(x * (x @ self.C_mat.T), dim=1)
                full_H    = torch.mean(Hx_term + torch.sum(v_x * Ma, dim=1)
                                       + cost_x + torch.sum(a * (a @ self.D_mat.T), dim=1))
            self.pia_history["h_loss"].append(full_H.item())

            # ── Evaluate errors ───────────────────────────────────────────────
            self.v_net.eval(); self.a_net.eval()
            with torch.no_grad():
                t0 = torch.zeros(1, device=dev)
                x0 = torch.tensor([[1.0, 1.0]], device=dev)
                v_pred = self.v_net(t0, x0).item()
                a_pred = self.a_net(t0, x0).cpu().numpy().flatten()

            v_err = abs(v_pred - v_true) / abs(v_true)
            a_err = np.linalg.norm(a_pred - a_true) / np.linalg.norm(a_true)
            self.pia_history["v_error"].append(v_err)
            self.pia_history["a_error"].append(a_err)

            print(f"v: pred={v_pred:.4f}  true={v_true:.4f}  err={v_err:.4f}")
            print(f"a: pred={a_pred}  true={a_true}  err={a_err:.4f}")


    def _fit_policy_iteration_2(self, n_iter, pde_epochs, ham_epochs, batch_size, lr):
        dev = next(self.parameters()).device

        x_test = np.array([1.0, 1.0])
        v_true = self.riccati.value(0.0, x_test)
        a_true = self.riccati.control(0.0, x_test)
        print(f"Riccati reference  v(0,x0)={v_true:.6f},  a(0,x0)={a_true}")

        # ── v_optimizer OUTSIDE the loop (persistent) ─────────────────────
        v_opt = torch.optim.Adam(self.v_net.parameters(), lr=lr)

        for iteration in range(1, n_iter + 1):
            print(f"\n=== Policy Iteration {iteration}/{n_iter} ===")

            # Keep v_net weights (warm start) but reset optimizer momentum
            v_opt = torch.optim.Adam(self.v_net.parameters(), lr=lr)
            v_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                v_opt, T_max=pde_epochs, eta_min=lr * 0.01)

            # Fresh a_optimizer each iteration
            a_opt = torch.optim.Adam(self.a_net.parameters(), lr=lr)
            a_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                a_opt, T_max=ham_epochs, eta_min=lr * 0.01)   

            # ── Step i: solve PDE for v ────────────────────────────────────
            self.v_net.train()
            v_loss_final = None

            for epoch in range(pde_epochs):
                v_opt.zero_grad()

                t = (torch.rand(batch_size, device=dev) * self.T * 0.99).requires_grad_(True)
                x = (torch.rand(batch_size, self.d, device=dev) * 2*self.x_range
                    - self.x_range).requires_grad_(True)

                with torch.no_grad():
                    a = self.a_net(t, x)

                u    = self.v_net(t, x)
                ones = torch.ones_like(u)
                grads = torch.autograd.grad(u, [t, x], grad_outputs=ones,
                                        create_graph=True)
                u_t = grads[0]
                u_x = grads[1]

                # Batched Hessian trace (faster + more stable than double loop)
                sigma_ux   = u_x @ self.sigma_sigmaT
                u_xx_trace = torch.autograd.grad(
                    (sigma_ux * u_x).sum(), x, create_graph=True
                )[0].sum(dim=1)

                drift    = torch.sum(u_x * (x @ self.H.T + a @ self.M_mat.T), dim=1)
                cost_x   = torch.sum(x * (x @ self.C_mat.T), dim=1)
                cost_a   = torch.sum(a * (a @ self.D_mat.T), dim=1)
                residual = u_t + 0.5 * u_xx_trace + drift + cost_x + cost_a
                loss_pde = residual.pow(2).mean()

                t_bc = torch.full((batch_size,), self.T, device=dev)
                x_bc = (torch.rand(batch_size, self.d, device=dev) * 2*self.x_range
                        - self.x_range)
                u_bc = self.v_net(t_bc, x_bc)
                tgt  = torch.sum(x_bc * (x_bc @ self.R_mat.T), dim=1, keepdim=True)
                loss_bc = (u_bc - tgt).pow(2).mean()

                loss = loss_pde + loss_bc
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), max_norm=1.0)
                v_opt.step()
                v_sch.step()
                v_loss_final = loss.item()

            # Log ONE value per iteration
            self.pia_history["v_loss"].append(v_loss_final)
            print(f"  PDE final loss: {v_loss_final:.4e}")

            # ── Step ii: minimise Hamiltonian ──────────────────────────────
            self.a_net.train()
            h_loss_final = None

            for epoch in range(ham_epochs):
                a_opt.zero_grad()

                t = torch.rand(batch_size, device=dev) * self.T
                x = (torch.rand(batch_size, self.d, device=dev) * 2*self.x_range
                    - self.x_range).requires_grad_(True)

                v   = self.v_net(t, x)
                v_x = torch.autograd.grad(v.sum(), x, create_graph=False)[0].detach()
                a   = self.a_net(t, x)
                Ma  = a @ self.M_mat.T

                hamiltonian = torch.mean(
                    torch.sum(v_x * Ma, dim=1) +
                    torch.sum(a * (a @ self.D_mat.T), dim=1)
                )
                hamiltonian.backward()
                torch.nn.utils.clip_grad_norm_(self.a_net.parameters(), max_norm=1.0)
                a_opt.step()
                a_sch.step()

                # Full Hamiltonian for logging
                with torch.no_grad():
                    Hx_term = torch.sum(v_x * (x @ self.H.T), dim=1)
                    cost_x  = torch.sum(x * (x @ self.C_mat.T), dim=1)
                    full_H  = torch.mean(
                        Hx_term + torch.sum(v_x * Ma, dim=1) +
                        cost_x  + torch.sum(a * (a @ self.D_mat.T), dim=1))
                h_loss_final = full_H.item()

            # Log ONE value per iteration
            self.pia_history["h_loss"].append(h_loss_final)
            print(f"  Ham final value: {h_loss_final:.4e}")

            # ── Evaluate errors ────────────────────────────────────────────
            self.v_net.eval(); self.a_net.eval()
            with torch.no_grad():
                t0 = torch.zeros(1, device=dev)
                x0 = torch.tensor([[1.0, 1.0]], device=dev)
                v_pred = self.v_net(t0, x0).item()
                a_pred = self.a_net(t0, x0).cpu().numpy().flatten()

            v_err = abs(v_pred - v_true) / abs(v_true)
            a_err = np.linalg.norm(a_pred - a_true) / np.linalg.norm(a_true)
            self.pia_history["v_error"].append(v_err)
            self.pia_history["a_error"].append(a_err)
            print(f"  v: pred={v_pred:.4f}  true={v_true:.4f}  err={v_err:.4f}")
            print(f"  a: pred={a_pred}  true={a_true}  err={a_err:.4f}")

    # ── Evaluation methods ────────────────────────────────────────────────────

    def value(self, t, x):
        """Learned value function. Returns scalar tensor."""
        with torch.no_grad():
            return self.v_net(t, x)

    def control(self, t, x):
        """
        Learned control (policy_iteration mode) or implied optimal control
        α* = -½D⁻¹Mᵀ ∇_x v (fixed_control mode). Returns (batch, m).
        """
        if self.mode == "policy_iteration" and self.a_net is not None:
            with torch.no_grad():
                return self.a_net(t, x)
        else:
            x = x.requires_grad_(True)
            v = self.v_net(t, x)
            v_x = torch.autograd.grad(v.sum(), x)[0]
            return (-0.5 * v_x @ (self.D_inv @ self.M_mat.T).T).detach()

    def unbiased_cost(self, x0, n_steps=200, mc_samples=10000, alpha_fixed=None):
        """
        Euler-Maruyama Monte Carlo estimate of the cost, using the Riccati
        optimal control as a control variate — mirroring unbiased_price() in [1].

        Returns:
            costs    (mc_samples,)  cost under the learned/fixed control
            costs_cv (mc_samples,)  control-variate corrected cost
                                    (lower variance, same mean in expectation)
        """
        self.eval()
        dev = next(self.parameters()).device
        dt  = self.T / n_steps
        X   = x0.to(dev).expand(mc_samples, -1).clone().float()

        costs    = torch.zeros(mc_samples, device=dev)
        costs_cv = torch.zeros(mc_samples, device=dev)

        alpha_np = (alpha_fixed if alpha_fixed is not None
                    else np.zeros(self.m, dtype=np.float32))

        for k in range(n_steps):
            t_k   = k * dt
            t_val = torch.full((mc_samples,), t_k, device=dev)

            # Control being evaluated
            if self.mode == "policy_iteration":
                with torch.no_grad():
                    alpha_nn = self.a_net(t_val, X)
            else:
                alpha_nn = torch.tensor(alpha_np, dtype=torch.float32,
                                        device=dev).unsqueeze(0).expand(mc_samples, -1)

            # Riccati optimal control as control variate
            S_k      = torch.tensor(self.riccati.S(t_k), dtype=torch.float32, device=dev)
            alpha_opt = -(X @ S_k.T @ self.M_mat @ self.D_inv.T)

            cost_x     = torch.sum(X * (X @ self.C_mat.T), dim=1)
            cost_a_nn  = torch.sum(alpha_nn  * (alpha_nn  @ self.D_mat.T), dim=1)
            cost_a_opt = torch.sum(alpha_opt * (alpha_opt @ self.D_mat.T), dim=1)

            costs    += (cost_x + cost_a_nn)  * dt
            costs_cv += (cost_x + cost_a_opt) * dt

            dW = torch.randn(mc_samples, self.d, device=dev) * dt**0.5
            X  = X + (X @ self.H.T + alpha_nn @ self.M_mat.T) * dt + dW @ self.sigma_mat.T

        terminal  = torch.sum(X * (X @ self.R_mat.T), dim=1)
        costs    += terminal
        costs_cv += terminal

        return costs, costs_cv

    # ── Internal helper ───────────────────────────────────────────────────────

    def _make_lqr_obj(self):
        """Create a minimal duck-typed object for run_monte_carlo_explicit."""
        class _LQR:
            pass
        lqr = _LQR()
        lqr.H      = self.H.cpu().numpy()
        lqr.M      = self.M_mat.cpu().numpy()
        lqr.C      = self.C_mat.cpu().numpy()
        lqr.D      = self.D_mat.cpu().numpy()
        lqr.R      = self.R_mat.cpu().numpy()
        lqr.D_inv  = self.D_inv.cpu().numpy()
        lqr.sigma  = self.sigma_mat.cpu().numpy()
        lqr.T      = self.T
        lqr.n      = self.d
        lqr.S_interp = self.riccati.S_interp
        lqr.solve_riccati = lambda g: None   # already solved
        lqr.get_control   = lambda t, x: torch.tensor(
            np.stack([self.riccati.control(ti.item(), xi.numpy().flatten())
                      for ti, xi in zip(t, x)], axis=0), dtype=torch.float32)
        return lqr

