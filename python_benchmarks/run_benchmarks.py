import sys
import os

# Add problems directory to path if needed (implicit via import)
from problems.oscillating_masses import run_oscillating_masses
from problems.quadrotor_linear import run_quadrotor_linear
from problems.spacecraft import run_spacecraft

def main():
    print("Running Python Benchmarks (Reference Solvers)...")
    
    horizons = [15, 30, 50]
    
    for N in horizons:
        print(f"\n\n========================================")
        print(f"BENCHMARKING WITH HORIZON N={N}")
        print(f"========================================")
        
        try:
            run_oscillating_masses(N)
        except Exception as e: print(f"Oscillating Masses failed: {e}")
            
        try:
            run_quadrotor_linear(N)
        except Exception as e: print(f"Quadrotor Linear failed: {e}")
            
        try:
            run_spacecraft(N)
        except Exception as e: print(f"Spacecraft failed: {e}")
            
        try:
            from problems.triple_integrator import run_triple_integrator
            run_triple_integrator(N)
        except Exception as e: print(f"Triple Integrator failed: {e}")

    print("\nAll Python benchmarks completed successfully.")

if __name__ == "__main__":
    main()
