import matplotlib.pyplot as plt
import csv
import glob
import os
import re
import math
import numpy as np

def plot_benchmark_results():
    results_dir = "simulation_results"
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all result files
    files = glob.glob(os.path.join(results_dir, "results_*.csv"))
    if not files:
        print(f"No result files found in {results_dir}. Run ./test_suite and python benchmarks first.")
        return

    print(f"Found {len(files)} result files.")

    # Sort solvers by length desc to avoid partial matches (e.g. OSQP vs OSQP_Direct)
    known_solvers = ["MyADMM", "OCSS", 
                     "cvxpy_Clarabel", "cvxpy_OSQP", "cvxpy_ECOS", 
                     "CVXPYGEN",
                     "Clarabel", "OSQP_Direct", "OSQP", "DAQP_Direct"]
    known_solvers.sort(key=len, reverse=True)

    # Structure: problems[Problem][N][Solver] = filepath
    problems = {} 

    for filepath in files:
        filename = os.path.basename(filepath)
        # Format: results_{Solver}_{Problem}_N{N}.csv
        
        # Remove prefix/suffix
        if not filename.startswith("results_") or not filename.endswith(".csv"):
            continue
            
        content = filename[8:-4] # Remove results_ and .csv
        
        # Extract N
        # Look for last _N
        n_match = re.search(r"_N(\d+)$", content)
        if not n_match:
            print(f"Skipping {filename}: No N found")
            continue
            
        N = int(n_match.group(1))
        base_rest = content[:n_match.start()] # Solver_Problem
        
        # Identify Solver
        solver = "Unknown"
        problem = base_rest
        
        for s in known_solvers:
            if base_rest.startswith(s + "_"):
                solver = s
                problem = base_rest[len(s)+1:]
                break
        
        if problem not in problems: problems[problem] = {}
        if N not in problems[problem]: problems[problem][N] = {}
        problems[problem][N][solver] = filepath

    print(f"Identified {len(problems)} unique problems.")

    # Plotting Loop
    for problem, n_map in problems.items():
        sorted_ns = sorted(n_map.keys())
        print(f"\nProcessing Problem: {problem} (N={sorted_ns})")
        
        # 1. Performance Comparison (Time vs N)
        plot_performance_scaling(problem, n_map, output_dir)
        
        # 2. Trajectory Plots per N
        for N in sorted_ns:
            solvers_map = n_map[N]
            plot_trajectories(problem, N, solvers_map, output_dir)

def get_style(solver_name):
    styles = {
        'OCSS':            {'c': 'k',         'ls': '-',    'm': None, 'lw': 2.0},
        'MyADMM':          {'c': 'k',         'ls': '-',    'm': None, 'lw': 2.0},
        'OSQP_Direct':     {'c': 'r',         'ls': '--',   'm': 'x',  'lw': 1.5, 'markevery': 5},
        'OSQP':            {'c': 'b',         'ls': ':',    'm': 'o',  'lw': 1.5, 'ms': 4, 'markevery': 5},
        
        # Updated keys for Python benchmarks
        'cvxpy_OSQP':      {'c': 'tab:green', 'ls': '-.',   'm': '+',  'lw': 1.5, 'markevery': 5},
        'cvxpy_Clarabel':  {'c': 'tab:cyan',  'ls': '-.',   'm': '2',  'lw': 1.5, 'ms': 6, 'markevery': 5},
        'cvxpy_ECOS':      {'c': 'tab:orange','ls': ':',    'm': 'p',  'lw': 1.5, 'ms': 5, 'markevery': 5},
        
        'CVXPYGEN':        {'c': 'm',         'ls': '-',    'm': '*',  'lw': 1.0, 'markevery': 5, 'markersize': 4},
        'Clarabel':        {'c': 'm',         'ls': ':',    'm': 'd',  'lw': 1.5, 'ms': 4, 'markevery': 5},
        'DAQP_Direct':     {'c': 'y',         'ls': '--',   'm': '*',  'lw': 1.5, 'ms': 6, 'markevery': 5},
    }
    default_style = {'c': 'gray', 'ls': '--', 'm': '.', 'lw': 1.0}
    return styles.get(solver_name, default_style)

def plot_performance_scaling(problem, n_map, output_dir):
    # Aggregating data
    # solvers[Solver] = ([N1, N2...], [Time1, Time2...])
    solver_perf = {}
    
    for N, solvers_map in n_map.items():
        for solver, filepath in solvers_map.items():
            try:
                with open(filepath, 'r') as f:
                    line = f.readline()
                    if line.startswith("# Solve Time (ms):"):
                        try:
                            t = float(line.split(":")[1].strip())
                        except:
                            t = np.nan
                    else:
                        t = np.nan
                
                if solver not in solver_perf: solver_perf[solver] = ([], [])
                solver_perf[solver][0].append(N)
                solver_perf[solver][1].append(t)
            except Exception as e:
                print(f"Error reading time from {filepath}: {e}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for solver, (ns, times) in solver_perf.items():
        # Sort by N
        zipped = sorted(zip(ns, times))
        ns_sorted, times_sorted = zip(*zipped)
        
        st = get_style(solver)
        ax.plot(ns_sorted, times_sorted, label=solver, 
                color=st['c'], marker=st['m'], linestyle=st['ls'], linewidth=st['lw'])
        
    ax.set_xlabel("Horizon Length (N)")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title(f"Performance Scaling: {problem}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log') # Log scale often useful for benchmarking
    
    out_file = os.path.join(output_dir, f"Performance_{problem}.png")
    fig.savefig(out_file, dpi=100)
    print(f"Saved Performance Plot: {out_file}")
    plt.close(fig)

def plot_trajectories(problem, N, solvers_map, output_dir):
    try:
        # Determine nx/nu by peeking
        first_filepath = next(iter(solvers_map.values()))
        with open(first_filepath, 'r') as f:
            line1 = f.readline()
            if line1.startswith("#"):
                headers = next(csv.reader(f))
            else:
                f.seek(0)
                headers = next(csv.reader(f))
        
        x_cols_peek = [h for h in headers if h.startswith('x')]
        u_cols_peek = [h for h in headers if h.startswith('u')]
        nx = len(x_cols_peek)
        nu = len(u_cols_peek)

        # Grid setup
        total_plots = nx + nu
        n_cols = 3
        n_rows = math.ceil(total_plots / n_cols)
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows), sharex=True)
        if n_rows * n_cols == 1: axs = np.array([axs])
        axs = axs.flatten()

        for solver_name, filepath in solvers_map.items():
            times = []
            x_data = [[] for _ in range(nx)]
            u_data = [[] for _ in range(nu)]
            solve_time = None
            
            with open(filepath, 'r') as csvfile:
                first_line = csvfile.readline()
                if first_line.startswith("# Solve Time (ms):"):
                    try:
                        solve_time = float(first_line.split(":")[1].strip())
                    except: pass
                    reader = csv.reader(csvfile)
                    headers = next(reader)
                else:
                    csvfile.seek(0)
                    reader = csv.reader(csvfile)
                    headers = next(reader)
                    
                x_cols_indices = [i for i, h in enumerate(headers) if h.startswith('x')]
                u_cols_indices = [i for i, h in enumerate(headers) if h.startswith('u')]
                
                for row in reader:
                    times.append(float(row[0]))
                    for i, idx in enumerate(x_cols_indices):
                        x_data[i].append(float(row[idx]))
                    for i, idx in enumerate(u_cols_indices):
                        u_data[i].append(float(row[idx]))

            # Check for failure (NaNs)
            failed = False
            for d in x_data + u_data:
                if np.any(np.isnan(d)) or np.any(np.isinf(d)):
                    failed = True; break
            
            st = get_style(solver_name)
            label_suffix = f" ({solve_time:.2f} ms)" if solve_time is not None else ""
            display_name = solver_name.replace("CVXPY_", "cvxpy_")
            final_label = f"{display_name}{label_suffix}"
            if failed: final_label += " (Failed)"

            # Plot States
            for i in range(nx):
                ax_idx = i
                if ax_idx < len(axs):
                    axs[ax_idx].plot(times, x_data[i], 
                            color=st.get('c'), ls=st.get('ls'), marker=st.get('m'),
                            ms=st.get('ms', 4), lw=st.get('lw'), alpha=0.8,
                            label=final_label if i==0 else None)

            # Plot Inputs
            for i in range(nu):
                ax_idx = nx + i
                if ax_idx < len(axs):
                    if st.get('ls') == 'None' or st.get('ls') is None:
                         axs[ax_idx].plot(times, u_data[i], 
                                    color=st.get('c'), ls='None', marker=st.get('m'),
                                    ms=st.get('ms', 6), alpha=0.8,
                                    label=final_label if i==0 else None)
                    else:
                         axs[ax_idx].step(times, u_data[i], where='post',
                                    color=st.get('c'), ls=st.get('ls'), lw=st.get('lw'),
                                    label=final_label if i==0 else None)

        # Finalize
        fig.suptitle(f'{problem} (Horizon N={N})', fontsize=16)
        for i, ax in enumerate(axs):
            if i < nx:
                ax.set_title(f'State x[{i}]', fontsize=10)
                ax.grid(True, alpha=0.3)
                if i == 0: ax.legend(loc='upper right', framealpha=0.9, fontsize='small')
            elif i < nx + nu:
                u_idx = i - nx
                ax.set_title(f'Input u[{u_idx}]', fontsize=10)
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')

        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        out_file = os.path.join(output_dir, f"Traj_{problem}_N{N}.png")
        fig.savefig(out_file, dpi=100)
        print(f"Saved Trajectory Plot: {out_file}")
        plt.close(fig)

    except Exception as e:
        print(f"Error plotting {problem} N={N}: {e}")

if __name__ == "__main__":
    plot_benchmark_results()
