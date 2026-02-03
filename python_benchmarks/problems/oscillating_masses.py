import numpy as np
import cvxpy as cp
import sys
import os

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run_ocp

def run_oscillating_masses(N=30):
    num_masses = 4
    nx = 2 * num_masses
    nu = num_masses
    
    m = 1.0; k_spring = 1.0; d_damp = 0.1; dt = 0.5
    
    Ac = np.zeros((nx, nx))
    Bc = np.zeros((nx, nu))
    
    Ac[0:num_masses, num_masses:2*num_masses] = np.eye(num_masses)
    
    K = np.zeros((num_masses, num_masses))
    for i in range(num_masses):
        K[i, i] = 2 * k_spring
        if i > 0: K[i, i-1] = -k_spring
        if i < num_masses-1: K[i, i+1] = -k_spring
    
    D = np.eye(num_masses) * d_damp
    
    Ac[num_masses:, 0:num_masses] = -(1.0/m) * K
    Ac[num_masses:, num_masses:] = -(1.0/m) * D
    
    Bc[num_masses:, :] = (1.0/m) * np.eye(num_masses)
    
    Ad = np.eye(nx) + Ac * dt
    Bd = Bc * dt
    
    Q = 2.0 * np.eye(nx)
    R = np.eye(nu)
    Pn = Q
    
    x0 = np.zeros(nx)
    for i in range(num_masses):
        x0[i] = 1.0 if i < num_masses/2 else -1.0
        
    # Bounds
    umin = np.full(nu, -1.0)
    umax = np.full(nu, 1.0)
    
    xmin = np.full(nx, -np.inf)
    xmax = np.full(nx, np.inf)
    xmin[:num_masses] = -4.0
    xmax[:num_masses] = 4.0
    
    def u_con(u, k): return [u >= umin, u <= umax]
    
    def x_con(x, k):
        # We can use the vector bounds directly in CVXPY too via list comprehension or broadcasting
        return [x >= xmin, x <= xmax]
        
    # Run CVXPY version
    run_ocp("Oscillating_Masses", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver=cp.OSQP)
    
    # Run Direct OSQP version
    from common import run_direct_osqp
    run_direct_osqp("Oscillating_Masses", N, nx, nu, Ad, Bd, Q, R, Pn, x0, umin, umax, xmin, xmax)

    # Run Direct DAQP version
    from common import run_direct_daqp
    run_direct_daqp("Oscillating_Masses", N, nx, nu, Ad, Bd, Q, R, Pn, x0, umin, umax, xmin, xmax)


    # Run (cp.CLARABEL)
    run_ocp("Oscillating_Masses", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver=cp.CLARABEL)

    # Run CVXPY ECOS version - Removed
    # run_ocp("Oscillating_Masses", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver=cp.ECOS)

    # Run CVXPYGEN version - Removed for now
    # run_ocp("Oscillating_Masses", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver="CVXPYGEN")

if __name__ == "__main__":
    run_oscillating_masses()
