import numpy as np
import cvxpy as cp
import sys
import os

# Add parent directory to path to import common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run_ocp

def run_triple_integrator(N=20):
    print(f"Running Triple Integrator (3D) with N={N}...")
    
    # Dimensions
    # p (3), v (3), a (3) -> nx = 9
    # j (3) -> nu = 3
    nx = 9
    nu = 3
    dt = 0.1
    
    # Dynamics (Triple Integrator)
    # p+ = p + v*dt + 0.5*a*dt^2 + 1/6*j*dt^3
    # v+ = v + a*dt + 0.5*j*dt^2
    # a+ = a + j*dt
    
    Ad = np.eye(nx)
    Bd = np.zeros((nx, nu))
    
    # Fill Ad
    # p-v interaction (index 0-2 to 3-5)
    Ad[0:3, 3:6] = np.eye(3) * dt
    # p-a interaction (index 0-2 to 6-8)
    Ad[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
    # v-a interaction (index 3-5 to 6-8)
    Ad[3:6, 6:9] = np.eye(3) * dt
    
    # Fill Bd
    # p-j (0-2)
    Bd[0:3, :] = np.eye(3) * (1.0/6.0) * dt**3
    # v-j (3-5)
    Bd[3:6, :] = np.eye(3) * 0.5 * dt**2
    # a-j (6-8)
    Bd[6:9, :] = np.eye(3) * dt
    
    # Costs
    Q = np.eye(nx) * 1.0
    R = np.eye(nu) * 0.1
    Pn = Q
    
    # Initial State
    # Start at rest at -2, go to 0
    x0 = np.zeros(nx)
    x0[0:3] = -2.0
    
    # Constraints
    v_max = 5.0
    a_max = 10.0
    j_max = 20.0 # Input box constraint
    
    # Input Constraints (Box)
    umin = -np.ones(nu) * j_max
    umax = np.ones(nu) * j_max
    
    def u_con(u, k):
        return [u >= umin, u <= umax]
    
    def x_con(x, k):
        # x = [p, v, a]
        v = x[3:6]
        a = x[6:9]
        cons = []
        # Ball constraints
        cons.append(cp.norm(v, 2) <= v_max)
        cons.append(cp.norm(a, 2) <= a_max)
        return cons
        
    # Reference (Regulation to zero)
    xr = np.zeros(nx)
    
    # Run Clarabel
    run_ocp("Triple_Integrator", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver=cp.CLARABEL, xr=xr)
    
    # Run ECOS
    run_ocp("Triple_Integrator", N, nx, nu, Ad, Bd, Q, R, Pn, x0, x_con, u_con, solver=cp.ECOS, xr=xr)
    
    # OCSS will be run via C++

if __name__ == "__main__":
    run_triple_integrator()
