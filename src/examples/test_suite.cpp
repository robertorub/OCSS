#include <exception>
#include <iostream>

// Benchmark Definitions
#include "benchmarks/oscillating_masses.hpp"
#include "benchmarks/quadrotor_box.hpp"
#include "benchmarks/spacecraft.hpp"
#include "benchmarks/triple_integrator.hpp"

int main() {
  try {
    // Quadrotor
    test_linear_quadrotor<15>();
    test_linear_quadrotor<30>();
    test_linear_quadrotor<50>();

    // Oscillating Masses
    test_oscillating_masses<15>();
    test_oscillating_masses<30>();
    test_oscillating_masses<50>();

    // Spacecraft
    test_spacecraft<15>();
    test_spacecraft<30>();
    test_spacecraft<50>();

    // Triple Integrator
    test_triple_integrator<15>();
    test_triple_integrator<30>();
    test_triple_integrator<50>();
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
