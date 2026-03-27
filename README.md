# Stochastic Control & Dynamic Asset Allocation (SCDAA)
Aaro Parkkinen (s2102444), Conor Grant (s2890252) 
Contribution: Equal (50% / 50%) 

## Overview: 
This repository contains the Python implementation for solving a $2 \times 2$ Linear Quadratic Regulator (LQR) problem using Policy Iteration and the Deep Galerkin Method (DGM).Requirements

To run this code, you must have the following libraries installed: numpy, scipy, matplotlib, torch (PyTorch)

## Project Structure & Reproducibility
To reproduce the figures and tables included in the PDF report, run the following scripts:

Figure 1 & 2 lqr_mc_verification.py - Runs the Riccati solver and Monte Carlo convergence tests (Exercise 1.2).  
Figure 3 & 4 supervised_learning.py - Trains NNs for the value function and Markov control (Exercise 2.1 & 2.2).  
Figure 5 & 6 dgm_linear_pde.py - Solves the linear PDE for a constant control using DGM (Exercise 3.1).  
Figure 7 & 8 policy_iteration_main.py - Runs the full Policy Iteration Algorithm (PIA) and outputs convergence plots (Exercise 4.1).

Clone the repository: git clone https://github.com/Conorgrant45/Stochastic-Control-Dynamic-Asset-Allocation.git

#### Execute: Modular code that uses LQR.py with the main mechanics and report_file.ipynb for each phase of the report

Note: Some of the plots take several minutes to reproduce, mainly figure 2 of varying monte carlo samples

## Implementation Details
Riccati Solver: Uses scipy.integrate.solve_ivp with the RK45 method for high-precision benchmarks.  
Discretization: SDE trajectories are generated using an Implicit Euler-Maruyama scheme for superior numerical stability.  
Neural Networks: Implementation utilizes torch.autograd for exact spatial and temporal derivatives in the PDE residual.  
