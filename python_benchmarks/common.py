import cvxpy as cp
import numpy as np
import scipy.sparse as sparse
import osqp
import time
import os
import sys
import csv

try:
    from cvxpygen import cpg
except ImportError:
    cpg = None

try:
    import daqp
    from ctypes import c_double, c_int
except ImportError:
    daqp = None


def export_to_csv(solver_name, problem_name, x, u, N, solve_time_ms=0.0):
    """
    Exports trajectory data to a CSV file.
    
    Args:
        solver_name (str): Name of the solver.
        problem_name (str): Name of the problem.
        x (np.ndarray): State trajectory (nx, N+1).
        u (np.ndarray): Input trajectory (nu, N).
        N (int): Horizon length.
        solve_time_ms (float): Solve time in milliseconds.
    """
    # Format name: results_<Solver>_<Problem>.csv
    safe_problem_name = problem_name.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"simulation_results/results_{solver_name}_{safe_problem_name}_N{N}.csv"
    
    # Extract values if CVXPY variables
    x_val = x.value if hasattr(x, 'value') and x.value is not None else x
    u_val = u.value if hasattr(u, 'value') and u.value is not None else u

    try:
        with open(filename, 'w', newline='') as csvfile:
            # Write comment line manually before using csv writer for data
            csvfile.write(f"# Solve Time (ms): {solve_time_ms}\n")
            
            writer = csv.writer(csvfile)
            # Create headers
            headers = ['k']
            # Handle x/u shapes (numpy arrays from direct solvers might be 1D flat or 2D)
            # We assume inputs x and u are matrices (nx, N+1) and (nu, N) for consistency in this helper
            nx = x_val.shape[0] if len(x_val.shape) > 1 else x_val.shape[0]
            nu = u_val.shape[0] if len(u_val.shape) > 1 else u_val.shape[0]
            
            for i in range(nx): headers.append(f"x{i}")
            for i in range(nu): headers.append(f"u{i}")
            
            writer.writerow(headers)
            
            for k in range(N+1):
                row = [k]
                for i in range(nx): 
                    val = x_val[i, k] if len(x_val.shape)>1 else x_val[i]
                    row.append(val)
                
                if k < N:
                    for i in range(nu): 
                        val = u_val[i, k] if len(u_val.shape)>1 else u_val[i]
                        row.append(val)
                else:
                    for i in range(nu): row.append(0.0)
                writer.writerow(row)
        print(f"Trajectory saved to {filename}")
    except Exception as e:
        print(f"Failed to save CSV {filename}: {e}")

def run_direct_osqp(name, N, nx, nu, Ad, Bd, Q, R, Pn, x0, umin, umax, xmin, xmax, xr=None):
    """
    Runs a direct OSQP solver for an MPC problem.
    
    Args:
        name (str): Problem name.
        N (int): Horizon.
        nx (int): State dimension.
        nu (int): Input dimension.
        Ad, Bd (np.ndarray): Dynamics matrices.
        Q, R, Pn (np.ndarray): Cost matrices.
        x0 (np.ndarray): Initial state.
        umin, umax (np.ndarray): Input bounds.
        xmin, xmax (np.ndarray): State bounds.
        xr (np.ndarray, optional): Reference state.
    """
    print(f"\n========================================")
    print(f"DIRECT SOLVER TEST: {name} (OSQP)")
    print(f"========================================")
    
    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    
    if xr is None: xr = np.zeros(nx)
    
    # Construct P (Hessian)
    # Block diagonal: Q, Q, ..., Pn, R, R, ...
    qs = [Q] * N + [Pn]
    rs = [R] * N
    P = sparse.block_diag(qs + rs, format='csc')
    
    # Linear term q
    # qx = -Q*xr for k=0..N-1, -Pn*xr for k=N
    qx = []
    for k in range(N): qx.append(-Q @ xr)
    qx.append(-Pn @ xr)
    qx = np.hstack(qx)
    
    # qu = 0
    qu = np.zeros(N*nu)
    q = np.hstack([qx, qu])
    
    # Constraints A
    
    I_nx = sparse.eye(nx)
    Ax_blocks = []
    # Row 0
    Ax_row0 = [I_nx] + [None]*N
    Ax_blocks.append(Ax_row0)
    
    for k in range(N):
        # Row k+1
        # 0 ... -Ad (pos k), I (pos k+1) ...
        row = [None]*(N+1)
        row[k] = -Ad
        row[k+1] = I_nx
        Ax_blocks.append(row)
        
    Ax = sparse.bmat(Ax_blocks, format='csc')
    
    # Bu:
    # Row 0: 0
    # Row k+1: -Bd (pos k)
    
    Bu_blocks = []
    # Row 0: Needs explicit zero to define shape if all others are None
    # x(0) = x_init depends only on x, not u. So Bu row 0 is all zeros.
    # explicit zero block at (0,0)
    row0 = [sparse.csc_matrix((nx, nu))] + [None]*(N-1)
    Bu_blocks.append(row0)
    for k in range(N):
        row = [None]*N
        row[k] = -Bd
        Bu_blocks.append(row)
        
    Bu = sparse.bmat(Bu_blocks, format='csc')
    
    A_dyn = sparse.hstack([Ax, Bu])
    l_dyn = np.hstack([x0, np.zeros(N*nx)])
    u_dyn = l_dyn # Equality
    
    # 2. Bound constraints
    # xmin <= x <= xmax
    # umin <= u <= umax
    
    # Identity matrix for all vars
    A_bounds = sparse.eye((N+1)*nx + N*nu, format='csc')
    
    # x bounds
    l_x = np.kron(np.ones(N+1), xmin)
    u_x = np.kron(np.ones(N+1), xmax)
    
    # u bounds
    l_u = np.kron(np.ones(N), umin)
    u_u = np.kron(np.ones(N), umax)
    
    l_bounds = np.hstack([l_x, l_u])
    u_bounds = np.hstack([u_x, u_u])
    
    # Stack all
    A = sparse.vstack([A_dyn, A_bounds], format='csc')
    l = np.hstack([l_dyn, l_bounds])
    u = np.hstack([u_dyn, u_bounds])
    
    # Solve
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-4, eps_rel=1e-4) # Matching opts
    
    t0 = time.time()
    res = prob.solve()
    t1 = time.time()
    
    solve_time = res.info.solve_time * 1000.0 # ms
    print(f"Solve Time: {solve_time:.4f} ms")
    print(f"Optimal Cost: {res.info.obj_val}")
    
    # Extract results
    # Vars: x0...xN (size (N+1)*nx), u0...uN-1 (size N*nu)
    x_sol = res.x[:(N+1)*nx].reshape((N+1, nx)).T # (nx, N+1)
    u_sol = res.x[(N+1)*nx:].reshape((N, nu)).T   # (nu, N)
    
    if u_sol.size > 0:
        print(f" u[0] = {u_sol[:, 0]}")
        export_to_csv("OSQP_Direct", name, x_sol, u_sol, N, solve_time)
    else:
        print("Solver failed.")


def run_direct_daqp(name, N, nx, nu, Ad, Bd, Q, R, Pn, x0, umin, umax, xmin, xmax, xr=None):
    """
    Runs a direct DAQP solver for an MPC problem.
    
    Args:
        name (str): Problem name.
        N (int): Horizon.
        nx (int): State dimension.
        nu (int): Input dimension.
        Ad, Bd (np.ndarray): Dynamics matrices.
        Q, R, Pn (np.ndarray): Cost matrices.
        x0 (np.ndarray): Initial state.
        umin, umax (np.ndarray): Input bounds.
        xmin, xmax (np.ndarray): State bounds.
        xr (np.ndarray, optional): Reference state.
    """
    if daqp is None:
        print("DAQP python package not installed. Skipping direct DAQP test.")
        return

    print(f"\n========================================")
    print(f"DIRECT SOLVER TEST: {name} (DAQP)")
    print(f"========================================")
    
    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    
    if xr is None: xr = np.zeros(nx)
    
    # Construct H (Hessian) - same as P in OSQP
    qs = [Q] * N + [Pn]
    rs = [R] * N
    H = sparse.block_diag(qs + rs, format='csc').toarray().astype(c_double) 
    # Add small regularization to avoid singularity if H is only PSD
    H += np.eye(H.shape[0]) * 1e-6

    # Note: DAQP supports sparse but for safety and guaranteed c_double alignment with the provided example 
    # we convert to dense np.array with c_double. If performance is critical we can try sparse later.
    
    # Linear term f - same as q in OSQP
    qx = []
    for k in range(N): qx.append(-Q @ xr)
    qx.append(-Pn @ xr)
    qx = np.hstack(qx)
    qu = np.zeros(N*nu)
    f = np.hstack([qx, qu]).astype(c_double)
    
    # Constraints A (Dynamics + Bounds)
    # 1. Dynamics
    I_nx = sparse.eye(nx)
    Ax_blocks = []
    Ax_blocks.append([I_nx] + [None]*N) # Row 0
    for k in range(N):
        row = [None]*(N+1)
        row[k] = -Ad
        row[k+1] = I_nx
        Ax_blocks.append(row)
    Ax = sparse.bmat(Ax_blocks, format='csc')
    
    Bu_blocks = []
    row0 = [sparse.csc_matrix((nx, nu))] + [None]*(N-1)
    Bu_blocks.append(row0)
    for k in range(N):
        row = [None]*N
        row[k] = -Bd
        Bu_blocks.append(row)
    Bu = sparse.bmat(Bu_blocks, format='csc')
    
    A_dyn = sparse.hstack([Ax, Bu])
    l_dyn = np.hstack([x0, np.zeros(N*nx)])
    u_dyn = l_dyn # Equality
    
    # 2. Bound constraints
    A_bounds = sparse.eye((N+1)*nx + N*nu, format='csc')
    
    l_x = np.kron(np.ones(N+1), xmin)
    u_x = np.kron(np.ones(N+1), xmax)
    l_u = np.kron(np.ones(N), umin)
    u_u = np.kron(np.ones(N), umax)
    
    l_bounds = np.hstack([l_x, l_u])
    u_bounds = np.hstack([u_x, u_u])
    
    # Stack all
    A_sparse = sparse.vstack([A_dyn, A_bounds], format='csc')
    A = A_sparse.toarray().astype(c_double)
    
    blower = np.hstack([l_dyn, l_bounds]).astype(c_double)
    bupper = np.hstack([u_dyn, u_bounds]).astype(c_double)
    
    # Sense
    # 0 = Inequality (default)
    # We rely on blower == bupper to imply equality for dynamics.
    # Alternatively, we could set sense=5 (Active+Immutable?) for dynamics rows.
    # But usually standard QP solvers handle l=u fine.
    sense = np.zeros(len(blower), dtype=c_int)
    
    t0 = time.time()
    (x_res, fval, exitflag, info) = daqp.solve(H, f, A, bupper, blower, sense)
    t1 = time.time()
    
    solve_time = info['solve_time'] * 1000.0 if 'solve_time' in info else (t1-t0)*1000.0
    print(f"Solve Time: {solve_time:.4f} ms")
    print(f"Optimal Cost: {fval}")
    
    if exitflag == 1:
        # Extract results
        x_sol_raw = x_res.flatten()
        x_sol = x_sol_raw[:(N+1)*nx].reshape((N+1, nx)).T
        u_sol = x_sol_raw[(N+1)*nx:].reshape((N, nu)).T
        
        if u_sol.size > 0:
            print(f" u[0] = {u_sol[:, 0]}")
            export_to_csv("DAQP_Direct", name, x_sol, u_sol, N, solve_time)
    else:
        print(f"Solver failed: Exit flag {exitflag}")


def run_ocp(name, N, nx, nu, A, B, Q, R, Pn, x0, state_constr_fn=None, input_constr_fn=None, solver=cp.OSQP, xr=None):
    """
    Runs an Optimal Control Problem using CVXPY.
    
    Args:
        name (str): Problem name.
        N (int): Horizon.
        nx (int): State dimension.
        nu (int): Input dimension.
        A, B (np.ndarray): Dynamics matrices (Ad, Bd).
        Q, R, Pn (np.ndarray): Cost matrices.
        x0 (np.ndarray): Initial state.
        state_constr_fn (callable, optional): function(x_k, k) returning list of constraints.
        input_constr_fn (callable, optional): function(u_k, k) returning list of constraints.
        solver: CVXPY solver constant or string (e.g. cp.OSQP).
        xr (np.ndarray, optional): Reference state.
    """
    print(f"\n========================================")
    print(f"REFERENCE TEST: {name}")
    print(f"========================================")
    
    p_name_safe = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    
    # Solver parsing logic
    cp_solver = None
    cpg_solver_str = 'OSQP'
    is_cpg = False
    
    if isinstance(solver, str):
        if solver.startswith("CVXPYGEN"):
            is_cpg = True
            parts = solver.split("_")
            if len(parts) > 1:
                cpg_solver_str = parts[1]
                # cpg expects 'OSQP', 'SCS', 'ECOS', 'CLARABEL' ?
                # Standard is usually 'OSQP'.
                p_name_safe += f"_{cpg_solver_str.lower()}"
            else:
                cpg_solver_str = 'OSQP'
        else:
             # Try to resolve string to solver
             try:
                 cp_solver = getattr(cp, solver)
             except AttributeError:
                 pass
    else:
        cp_solver = solver

    
    # Named variables for CVXPYGEN
    x = cp.Variable((nx, N+1), name='x')
    u = cp.Variable((nu, N), name='u')
    
    # Dummy parameter to satisfy CVXPYGEN requirement (it fails if no params)
    dummy_param = cp.Parameter(name='dummy_param')
    dummy_param.value = 0.0
    
    cost = dummy_param
    constraints = [x[:, 0] == x0]
    
    # Handle reference xr if provided
    def get_xr():
        if xr is None: return np.zeros(nx)
        return xr

    ref = get_xr()

    for k in range(N):
        constraints.append(x[:, k+1] == A @ x[:, k] + B @ u[:, k])
        cost += 0.5*cp.quad_form(x[:, k] - ref, Q) + 0.5*cp.quad_form(u[:, k], R)
        
        if input_constr_fn:
            con = input_constr_fn(u[:, k], k)
            if con: constraints.extend(con)
            
        if state_constr_fn:
            con = state_constr_fn(x[:, k+1], k)
            if con: constraints.extend(con)

    cost += 0.5*cp.quad_form(x[:, N] - ref, Pn)
         
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        t0 = time.time()
        
        solve_opts = {'verbose': False}
        if cp_solver == cp.OSQP:
             solve_opts.update({'eps_abs': 1e-4, 'eps_rel': 1e-4})
        elif cp_solver == cp.CLARABEL:
             solve_opts.update({'tol_gap_abs': 1e-4, 'tol_gap_rel': 1e-4})
        elif cp_solver == cp.ECOS:
             # ECOS tuning for stability
             solve_opts.update({'max_iters': 200, 'abstol': 1e-7, 'reltol': 1e-7, 'feastol': 1e-7})
             
        if is_cpg:
             if cpg is None:
                 print("CVXPYGEN not installed.")
                 return
             
             # Generate code
             try:
                 code_dir = f"cpg_code_{p_name_safe}"
                 # Ensure CWD is in path for import to work
                 if os.getcwd() not in sys.path:
                     sys.path.append(os.getcwd())
                 
                 print(f"Generating CPG code in {code_dir} (Solver: {cpg_solver_str})...")
                 # generate_code compiles the wrapper
                 cpg.generate_code(prob, code_dir=code_dir, solver=cpg_solver_str)
                 
                 # Import manually
                 import importlib
                 importlib.invalidate_caches()
                 cpg_module = importlib.import_module(f"{code_dir}.cpg_solver")
                  
                 prob.register_solve('cpg', cpg_module.cpg_solve)
                 solve_opts = {'method': 'cpg'}
             except Exception as e:
                 import traceback
                 traceback.print_exc()
                 print(f"CVXPYGEN Failed: {e}")
                 # Export failed results so it appears in legend
                 solver_name_final = f"CVXPYGEN_{cpg_solver_str}" if cpg_solver_str != 'OSQP' else "CVXPYGEN"
                 x_fail = np.full((nx, N+1), np.nan)
                 u_fail = np.full((nu, N), np.nan)
                 export_to_csv(solver_name_final, name, x_fail, u_fail, N, 0.0)
                 return
             
        if is_cpg:
             prob.solve(**solve_opts)
        else:
             prob.solve(solver=cp_solver, **solve_opts)

        t1 = time.time()
        
        solve_time = prob.solver_stats.solve_time * 1000.0 if prob.solver_stats and prob.solver_stats.solve_time else (t1-t0)*1000.0
        
        # Determine basic solver name
        s_name = str(solver)
        if cp_solver == cp.OSQP: s_name = "OSQP"
        elif cp_solver == cp.CLARABEL: s_name = "Clarabel"
        elif cp_solver == cp.ECOS: s_name = "ECOS"
        elif isinstance(solver, str): s_name = solver
        
        if is_cpg:
            solver_name = f"CVXPYGEN_{cpg_solver_str}" if cpg_solver_str != 'OSQP' else "CVXPYGEN"
        else:
            solver_name = f"cvxpy_{s_name}"
        
        print(f"Solve Time: {solve_time:.4f} ms")
        is_success = False
        if prob.value is not None:
             print(f"Optimal Cost: {prob.value}")
             if not np.isnan(prob.value) and not np.isinf(prob.value):
                 is_success = True
        
        if is_success:
             if u.value is not None:
                 u0_show = u.value[:, 0] if len(u.shape) > 1 else [u.value[0]]
                 print(f" u[0] = {u0_show[:4]}")
             export_to_csv(solver_name, name, x, u, N, solve_time)
        else:
             print("Result contains NaN/Inf or Solver failed. Exporting as Failed.")
             # Create NaN arrays
             x_fail = np.full((nx, N+1), np.nan)
             u_fail = np.full((nu, N), np.nan)
             export_to_csv(solver_name, name, x_fail, u_fail, N, solve_time)

    except Exception as e:
        print(f"Solver Error: {e}")

