import subprocess
import re
import statistics
import json
from collections import defaultdict

def run_benchmark(executable_path):
    """Runs the benchmark executable and returns the stdout."""
    result = subprocess.run([executable_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Benchmark failed with return code {result.returncode}")
    return result.stdout

def parse_output(output):
    """Parses the output to extract benchmark names and solve times."""
    # Pattern to match:
    # BENCHMARK: Linear Quadrotor (Box)
    # ...
    # Solve Time: 0.1537 ms
    # ...
    # Trajectory saved to simulation_results/results_OCSS_Linear_Quadrotor_N15.csv
    
    # We need to map the "BENCHMARK" title to the specific N value found in the filename line
    # or rely on the order if they are consistent.
    # Looking at the output, the blocks are sequential.
    
    benchmarks = []
    
    # Split by the separator lines
    blocks = output.split("========================================")
    
    current_benchmark_name = None
    
    for block in blocks:
        if not block.strip():
            continue
            
        # Extract benchmark name
        name_match = re.search(r"BENCHMARK: (.+)", block)
        if name_match:
            current_benchmark_name = name_match.group(1).strip()
            
        # Extract solve time
        time_match = re.search(r"Solve Time: ([\d\.]+) ms", block)
        
        # Extract N from filename (e.g. ..._N15.csv)
        n_match = re.search(r"_N(\d+)\.csv", block)
        
        if time_match and n_match:
            solve_time = float(time_match.group(1))
            n_val = int(n_match.group(1))
            if current_benchmark_name:
                full_name = f"{current_benchmark_name} (N={n_val})"
                benchmarks.append((full_name, solve_time))
                
    return benchmarks

def main():
    executable = "./build/test_suite"
    num_runs = 20
    
    print(f"Running benchmark {num_runs} times to collect statistics...")
    
    all_results = defaultdict(list)
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end="\r")
        try:
            output = run_benchmark(executable)
            results = parse_output(output)
            for name, time in results:
                all_results[name].append(time)
        except Exception as e:
            print(f"\nError during run {i+1}: {e}")
            break
            
    # Export to JSON
    # Parse name to extract Problem, Solver, N for structured usage if needed, 
    # but for now we can save the map directly and parse consistently in plotting script.
    # Format key: "ProblemName (SolverName) (N=X)" -> but C++ output format is different.
    # C++ Output parsed keys: "Linear Quadrotor (Box) (N=15)"
    
    # Let's verify the key format from parse_output:
    # "Linear Quadrotor (Box) (N=15)"
    
    json_data = []
    for full_name, times in all_results.items():
        # Simple parsing of the full_name to structured dict
        # Expected: "{Problem} ({Solver}) (N={N})"
        # But C++ output is: "{Problem} (N={N})" where Solver is implicitly OCSS/ADMM
        
        # We can just save the flat map and let the plotter handle parsing strings
        # Or better, parse here to be clean.
        
        # Regex for C++ output keys from collect_stats.py:
        # full_name = f"{current_benchmark_name} (N={n_val})"
        # current_benchmark_name example: "Linear Quadrotor (Box)"
        
        n_match = re.search(r"\(N=(\d+)\)", full_name)
        n_val = int(n_match.group(1)) if n_match else 0
        
        # Remove (N=..)
        base_name = full_name.replace(f"(N={n_val})", "").strip()
        
        # Assume Solver is "OCSS" for C++ benchmarks
        solver = "OCSS"
        problem = base_name
        
        json_data.append({
            "solver": solver,
            "problem": problem,
            "N": n_val,
            "times": times
        })
        
    with open("stats_cpp.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n\nSaved statistics to stats_cpp.json")
            
    print("\n\n" + "="*60)
    print(f"{'Benchmark Subproblem':<40} | {'Mean (ms)':<10} | {'StdDev':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 90)
    
    for name, times in all_results.items():
        if not times:
            continue
        avg = statistics.mean(times)
        try:
            stdev = statistics.stdev(times)
        except statistics.StatisticsError:
            stdev = 0.0
        min_val = min(times)
        max_val = max(times)
        
        print(f"{name:<40} | {avg:<10.4f} | {stdev:<10.4f} | {min_val:<10.4f} | {max_val:<10.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
