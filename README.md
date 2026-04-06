# Stochastic Control & Dynamic Asset Allocation (SCDAA)

**Authors:** Aaro Parkkinen (s2102444), Conor Grant (s2890252)  
**Contribution:** Equal (50% / 50%)

## Overview
This repository contains the Python implementation for solving a 2x2 Linear Quadratic 
Regulator (LQR) problem using Policy Iteration combined with the Deep Galerkin Method (DGM).

## Dependencies
Install the required libraries before running:

    pip install numpy scipy matplotlib torch

## How to Reproduce Results
All figures and results in the PDF report are produced by running the Jupyter notebook:

    jupyter notebook Assignment.ipynb

Run all cells top to bottom in a single session. The notebook is divided into sections 
matching the report:

| Section | Exercise | Output |
|---|---|---|
| Exercise 1.2 | LQR & Monte Carlo Verification | Figures 1 & 2 |
| Exercise 2.1 & 2.2 | Supervised Learning | Figures 3 & 4 |
| Exercise 3.1 | DGM for Linear PDE | Figures 5 & 6 |
| Exercise 4.1 | Policy Iteration Algorithm | Figures 7 & 8 + numerical table |

All figures are saved automatically to the figures/ directory.

## Project Structure
- LQR.py — Core implementation: LQR class, Monte Carlo simulators, neural network 
  architectures, DGM trainer, and Policy Iteration Algorithm
- dgm_lqr.py - PDE_DGM_LQR class
- Assignment.ipynb — Notebook to reproduce all figures and results in the report

## Runtime Warning
The Monte Carlo sample convergence plot (Figure 2) is the most computationally expensive step 
and may take 15-30 minute to run due to repeated simulation runs.

## Implementation Notes
- Riccati Solver: scipy.integrate.solve_ivp with RK45, rtol=1e-9, atol=1e-12
- SDE Discretisation: Implicit Euler-Maruyama scheme for numerical stability
- Neural Networks: PyTorch autograd for exact PDE derivatives without discretisation error
