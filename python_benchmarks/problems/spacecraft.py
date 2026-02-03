import numpy as np
import cvxpy as cp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run_ocp

def run_spacecraft(N=30):
    nx = 6; nu = 3
    n = 0.0011; dt = 30.0; m = 500.0
    
    Ac = np.zeros((nx, nx))
    Ac[0,3]=1; Ac[1,4]=1; Ac[2,5]=1
    Ac[3,0] = 3*n**2; Ac[3,4] = 2*n
    Ac[4,3] = -2*n
    Ac[5,2] = -n**2
    
    Bc = np.zeros((nx, nu))
    Bc[3:6, 0:3] = (1.0/m) * np.eye(3)
    
    Ad = np.eye(nx) + Ac * dt
    Bd = Bc * dt
    
    Q = np.zeros((nx, nx))
    np.fill_diagonal(Q, [1, 1, 1, 10, 10, 10])
    R = np.eye(nu)
    Pn = Q * 10.0
    
    x0 = np.zeros(nx)
    x0[0] = -100.0; x0[4] = 0.1
    
    
    # Constraints
    umin = np.full(nu, -1.0)
    umax = np.full(nu, 1.0)
    
    xmin = np.full(nx, -1000.0)
    xmax = np.full(nx, 1000.0)
    
    def u_con(u, k): return [u >= umin, u <= umax]
    def x_con(x, k): return [x >= xmin, x <= xmax]
    
    run_ocp("Spacecraft", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver=cp.OSQP)
    
    # Direct OSQP
    from common import run_direct_osqp
    run_direct_osqp("Spacecraft", N, nx, nu, Ad, Bd, Q, R, Pn, x0, umin, umax, xmin, xmax)

    # Direct DAQP
    from common import run_direct_daqp
    run_direct_daqp("Spacecraft", N, nx, nu, Ad, Bd, Q, R, Pn, x0, umin, umax, xmin, xmax)


    # Run (cp.CLARABEL)
    run_ocp("Spacecraft", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver=cp.CLARABEL)

    # Run CVXPY ECOS version
    run_ocp("Spacecraft", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver=cp.ECOS)

    # Run CVXPYGEN version - Removed for now
    # run_ocp("Spacecraft", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver="CVXPYGEN")

if __name__ == "__main__":
    run_spacecraft()
