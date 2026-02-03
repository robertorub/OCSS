import matplotlib.pyplot as plt
import json
import os
import numpy as np

def get_style(solver_name):
    styles = {
        'OCSS':            {'c': 'k',         'ls': '-',    'm': None, 'lw': 2.0},
        'MyADMM':          {'c': 'k',         'ls': '-',    'm': None, 'lw': 2.0},
        'OSQP_Direct':     {'c': 'r',         'ls': '--',   'm': 'x',  'lw': 1.5, 'markevery': 5},
        'OSQP':            {'c': 'b',         'ls': ':',    'm': 'o',  'lw': 1.5, 'ms': 4, 'markevery': 5},
        
        'cvxpy_OSQP':      {'c': 'tab:green', 'ls': '-.',   'm': '+',  'lw': 1.5, 'markevery': 5},
        'cvxpy_Clarabel':  {'c': 'tab:cyan',  'ls': '-.',   'm': '2',  'lw': 1.5, 'ms': 6, 'markevery': 5},
        'cvxpy_ECOS':      {'c': 'tab:orange','ls': ':',    'm': 'p',  'lw': 1.5, 'ms': 5, 'markevery': 5},
        
        'CVXPYGEN':        {'c': 'm',         'ls': '-',    'm': '*',  'lw': 1.0, 'markevery': 5, 'markersize': 4},
        'Clarabel':        {'c': 'm',         'ls': ':',    'm': 'd',  'lw': 1.5, 'ms': 4, 'markevery': 5},
        'DAQP_Direct':     {'c': 'y',         'ls': '--',   'm': '*',  'lw': 1.5, 'ms': 6, 'markevery': 5},
    }
    default_style = {'c': 'gray', 'ls': '--', 'm': '.', 'lw': 1.0}
    return styles.get(solver_name, default_style)

def clean_problem_name(name):
    # Standardize to Python style: "Linear_Quadrotor"
    name = name.replace(" (Box)", "").replace("(Box)", "").strip()
    name = name.replace(" (OSQP)", "").replace("(OSQP)", "").strip() # Remove (OSQP) suffix from Oscillating Masses
    name = name.replace(" (HCW)", "").replace("(HCW)", "").strip()
    name = name.replace(" (3D)", "").replace("(3D)", "").strip()
    name = name.replace(" ", "_")
    
    # Handle specific mappings if simple replacement isn't enough
    if "Linear_Quadrotor" in name: return "Linear_Quadrotor"
    if "Oscillating_Masses" in name: return "Oscillating_Masses"
    if "Spacecraft" in name: return "Spacecraft"
    if "Triple_Integrator" in name: return "Triple_Integrator"
    
    return name

def load_data():
    data = []
    
    # Load C++ Stats
    if os.path.exists("stats_cpp.json"):
        with open("stats_cpp.json", "r") as f:
            data.extend(json.load(f))
            
    # Load Python Stats
    if os.path.exists("stats_python.json"):
        with open("stats_python.json", "r") as f:
            data.extend(json.load(f))
            
    return data

def process_data(data):
    # Structure: problems[Problem][Solver][N] = {mean, min, max}
    problems = {}
    
    for entry in data:
        raw_prob = entry['problem']
        solver = entry['solver']
        N = entry['N']
        times = entry['times']
        
        if not times: continue
        
        prob = clean_problem_name(raw_prob)
        
        if prob not in problems: problems[prob] = {}
        if solver not in problems[prob]: problems[prob][solver] = {}
        
        problems[prob][solver][N] = {
            'mean': np.mean(times),
            'min': np.min(times),
            'max': np.max(times)
        }
        
    return problems

def plot_stats(problems, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    for problem, solvers_map in problems.items():
        print(f"Plotting Stats for {problem}...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Sort solvers to ensure consistent legend order or priority
        solvers = sorted(solvers_map.keys())
        
        for solver in solvers:
            n_map = solvers_map[solver]
            ns = sorted(n_map.keys())
            
            means = [n_map[n]['mean'] for n in ns]
            mins = [n_map[n]['min'] for n in ns]
            maxs = [n_map[n]['max'] for n in ns]
            
            st = get_style(solver)
            
            # Plot Mean Line
            ax.plot(ns, means, label=solver, 
                    color=st['c'], marker=st['m'], linestyle=st['ls'], linewidth=st['lw'])
            
            # Plot Corridor (Min-Max)
            ax.fill_between(ns, mins, maxs, color=st['c'], alpha=0.15)
            
        ax.set_xlabel("Horizon Length (N)", fontsize=12)
        ax.set_ylabel("Solve Time (ms)", fontsize=12)
        ax.set_title(f"Statistical Performance: {problem}\n(Mean with Min-Max Corridor)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Save
        safe_prob = problem.replace(" ", "_").replace("(", "").replace(")", "")
        out_file = os.path.join(output_dir, f"Stats_Performance_{safe_prob}.png")
        fig.savefig(out_file, dpi=150)
        print(f"Saved: {out_file}")
        plt.close(fig)

def main():
    data = load_data()
    if not data:
        print("No statistics data found. Run collect_stats.py and collect_python_stats.py first.")
        return
        
    problems = process_data(data)
    plot_stats(problems)

if __name__ == "__main__":
    main()
