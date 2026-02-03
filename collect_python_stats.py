import subprocess
import re
import statistics
import sys
import os
import json
from collections import defaultdict

def run_benchmark_script():
    """Runs the python benchmark script and returns the stdout."""
    # Use the current python executable (assuming it's the one in .venv if activated,
    # or we should explicitly use .venv/bin/python just to be safe if run from outside)
    
    python_exe = sys.executable
    script_path = "python_benchmarks/run_benchmarks.py"
    
    # Check if .venv exists in current directory, if so prefer it
    venv_python = ".venv/bin/python"
    if os.path.exists(venv_python):
        python_exe = venv_python

    cmd = [python_exe, script_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Stderr: {result.stderr}")
        raise Exception(f"Benchmark script failed with return code {result.returncode}")
    return result.stdout

def parse_output(output):
    """
    Parses the output to extract benchmark data.
    Returns a list of tuples: (full_name, solve_time_ms)
    """
    results = []
    
    current_N = None
    current_problem = None
    # We buffer solve time until we find the filename or solver confirmation
    pending_solve_time = None
    
    # Known problem names map to tidy names if needed, or use as is
    # In common.py: safe_problem_name = problem_name.replace(" ", "_")...
    # We can rely on the regex matching the filename structure: results_<Solver>_<Problem>_N<N>.csv
    
    lines = output.split('\n')
    for line in lines:
        line = line.strip()
        
        # 1. Detect Horizon
        m_hor = re.search(r"BENCHMARKING WITH HORIZON N=(\d+)", line)
        if m_hor:
            current_N = int(m_hor.group(1))
            continue
            
        # 2. Detect Solve Time
        m_time = re.search(r"Solve Time: ([\d\.]+) ms", line)
        if m_time:
            pending_solve_time = float(m_time.group(1))
            continue
            
        # 3. Detect Trajectory Save (contains Solver and Problem names)
        # Format: Trajectory saved to simulation_results/results_<Solver>_<Problem>_N<N>.csv
        if "Trajectory saved to" in line and pending_solve_time is not None:
             m_file = re.search(r"simulation_results/results_(.+)_N(\d+)\.csv", line)
             if m_file:
                 middle_part = m_file.group(1) # e.g. cvxpy_OSQP_Spacecraft or OSQP_Direct_Spacecraft
                 file_N = int(m_file.group(2))
                 
                 # Heuristic to split Solver and Problem
                 # We know the problems are: Linear_Quadrotor, Oscillating_Masses, Spacecraft, Triple_Integrator
                 known_problems = ["Linear_Quadrotor", "Oscillating_Masses", "Spacecraft", "Triple_Integrator"]
                 
                 found_prob = None
                 found_solver = None
                 
                 for prob in known_problems:
                     if middle_part.endswith(prob):
                         found_prob = prob
                         # Solver is everything before the problem
                         # middle_part = <Solver>_<Problem>
                         # Solver = middle_part[:-len(prob)-1] (minus 1 for the underscore)
                         found_solver = middle_part[:-(len(prob)+1)]
                         break
                 
                 if found_prob and found_solver:
                     # Construct full name
                     # e.g. Spacecraft (cvxpy_OSQP) (N=50)
                     full_name = f"{found_prob} ({found_solver}) (N={file_N})"
                     results.append((full_name, pending_solve_time))
                 else:
                     # Fallback if problem name not recognized or parsing failed
                     # just use the whole middle part
                     full_name = f"{middle_part} (N={file_N})"
                     results.append((full_name, pending_solve_time))
                 
                 # Reset pending time
                 pending_solve_time = None

    return results

def main():
    num_runs = 20
    print(f"Running Python benchmarks {num_runs} times to collect statistics...")
    
    all_results = defaultdict(list)
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end="\r")
        try:
            output = run_benchmark_script()
            run_data = parse_output(output)
            for name, time_val in run_data:
                all_results[name].append(time_val)
        except Exception as e:
            print(f"\nError during run {i+1}: {e}")
            break
            
    # Export to JSON
    json_data = []
    for full_name, times in all_results.items():
        # parse_output format: "{Problem} ({Solver}) (N={N})"
        # e.g. "Linear_Quadrotor (DAQP_Direct) (N=15)"
        
        n_match = re.search(r"\(N=(\d+)\)", full_name)
        n_val = int(n_match.group(1)) if n_match else 0
        
        # Remove (N=..)
        base = full_name.replace(f"(N={n_val})", "").strip()
        
        # Extract Solver inside parens
        # e.g. "Linear_Quadrotor (DAQP_Direct)"
        # Last parens block is solver
        s_match = re.search(r"\(([^)]+)\)$", base)
        if s_match:
            solver = s_match.group(1)
            # Remove solver from base to get problem
            # "Linear_Quadrotor (DAQP_Direct)" -> "Linear_Quadrotor "
            problem = base.replace(f"({solver})", "").strip()
        else:
            solver = "Unknown"
            problem = base
            
        json_data.append({
            "solver": solver,
            "problem": problem,
            "N": n_val,
            "times": times
        })
        
    with open("stats_python.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n\nSaved statistics to stats_python.json")
            
    print("\n\n" + "="*80)
    print(f"{'Benchmark Subproblem':<50} | {'Mean (ms)':<10} | {'StdDev':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 100)
    
    # Sort by name for nicer output
    sorted_names = sorted(all_results.keys())
    
    for name in sorted_names:
        times = all_results[name]
        if not times:
            continue
        avg = statistics.mean(times)
        try:
            stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        except statistics.StatisticsError:
            stdev = 0.0
        min_val = min(times)
        max_val = max(times)
        
        print(f"{name:<50} | {avg:<10.4f} | {stdev:<10.4f} | {min_val:<10.4f} | {max_val:<10.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
