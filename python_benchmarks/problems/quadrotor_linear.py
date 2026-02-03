import numpy as np
import scipy.sparse as sparse
import cvxpy as cp
import sys
import os
import time
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run_ocp, export_to_csv

def run_quadrotor_linear(N=30):
    print(f"Running Linear Quadrotor (CVXPY wrapper) with N={N}...")
    # Discrete time model of a quadcopter
    Ad = sparse.csc_matrix([
    [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
    [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
    [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
    [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
    [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
    [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
    [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
    [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
    [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
    ])
    Bd = sparse.csc_matrix([
    [0.,      -0.0726,  0.,     0.0726],
    [-0.0726,  0.,      0.0726, 0.    ],
    [-0.0152,  0.0152, -0.0152, 0.0152],
    [-0.,     -0.0006, -0.,     0.0006],
    [0.0006,   0.,     -0.0006, 0.0000],
    [0.0106,   0.0106,  0.0106, 0.0106],
    [0,       -1.4512,  0.,     1.4512],
    [-1.4512,  0.,      1.4512, 0.    ],
    [-0.3049,  0.3049, -0.3049, 0.3049],
    [-0.,     -0.0236,  0.,     0.0236],
    [0.0236,   0.,     -0.0236, 0.    ],
    [0.2107,   0.2107,  0.2107, 0.2107]])
    
    nx, nu = Bd.shape
    
    # Constraints
    u0 = 10.5916
    umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
    umax = np.array([13., 13., 13., 13.]) - u0
    xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
                    -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
    xmax = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
                    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    # Objective
    Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
    QN = Q
    R = 0.1*sparse.eye(4)
    xr = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

    x0 = np.zeros(12)
    
    # Custom run because logic inside run_ocp for xr is slightly different (run_ocp assumes xr=0 usually unless added support)
    # I added xr support to run_ocp in common.py!
    
    def u_con(u, k): return [u >= umin, u <= umax]
    def x_con(x, k): return [x >= xmin, x <= xmax]

    # Direct OSQP
    from common import run_direct_osqp, export_to_csv, run_direct_daqp
    run_direct_osqp("Linear_Quadrotor", N, nx, nu, Ad, Bd, Q, R, QN, x0, umin, umax, xmin, xmax, xr=xr)
    
    # Direct DAQP
    run_direct_daqp("Linear_Quadrotor", N, nx, nu, Ad, Bd, Q, R, QN, x0, umin, umax, xmin, xmax, xr=xr)
    
    # Run Clarabel (CVXPY)
    run_ocp("Linear_Quadrotor", N, nx, nu, Ad, Bd, Q, R, QN, x0, x_con, u_con, solver=cp.CLARABEL, xr=xr)

    # Run CVXPYGEN version - Removed for now
    # run_ocp("Linear_Quadrotor", N, nx, nu, Ad, Bd, Q, R, QN, x0, x_con, u_con, solver="CVXPYGEN", xr=xr)


if __name__ == "__main__":
    run_quadrotor_linear()
