#pragma once

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ocss/admm.hpp>
#include <string>
#include <vector>

// Helper to print test banner
inline void print_test_banner(const std::string &name) {
  std::cout << "\n========================================\n";
  std::cout << "BENCHMARK: " << name << "\n";
  std::cout << "========================================\n";
}

template <typename Solver, typename OCP, typename Vars>
void analyze_result(const Solver &solver, OCP &ocp, const Vars &vars,
                    double time_ms, const std::string &problem_name,
                    const std::string &solver_name = "OCSS") {
  double cost = 0.0;
  int N = vars.u.cols();

  // Use generic access if possible
  auto &Q = ocp.modify_Q();
  auto &R = ocp.modify_R();
  auto &Pn = ocp.modify_Pn();
  auto &q_vec = ocp.modify_q();
  auto &pn_vec = ocp.modify_pn();

  for (int k = 0; k < N; ++k) {
    cost += 0.5 * vars.x.col(k).dot(Q[k] * vars.x.col(k));
    cost += 0.5 * vars.u.col(k).dot(R[k] * vars.u.col(k));
    cost += q_vec.col(k).dot(vars.x.col(k));
  }
  cost += 0.5 * vars.x.col(N).dot(Pn * vars.x.col(N));
  cost += pn_vec.dot(vars.x.col(N));

  std::cout << "Computed Cost: " << std::fixed << std::setprecision(4) << cost
            << "\n";
  std::cout << "Solve Time: " << time_ms << " ms\n";
  int print_n = std::min((int)vars.u.rows(), 4);
  std::cout << "First Input u[0]: " << vars.u.col(0).head(print_n).transpose()
            << " ...\n";

  // EXPORT TO CSV
  // Format: results_<Solver>_<Problem>.csv
  std::string safe_problem_name = problem_name;
  std::replace(safe_problem_name.begin(), safe_problem_name.end(), ' ', '_');
  safe_problem_name.erase(
      std::remove(safe_problem_name.begin(), safe_problem_name.end(), '('),
      safe_problem_name.end());
  safe_problem_name.erase(
      std::remove(safe_problem_name.begin(), safe_problem_name.end(), ')'),
      safe_problem_name.end());

  std::string filename = "simulation_results/results_" + solver_name + "_" +
                         safe_problem_name + "_N" + std::to_string(N) + ".csv";

  std::ofstream file(filename);
  if (file.is_open()) {
    file << "# Solve Time (ms): " << time_ms << "\n";
    file << "k,";
    for (int i = 0; i < vars.x.rows(); ++i)
      file << "x" << i << ",";
    for (int i = 0; i < vars.u.rows(); ++i)
      file << "u" << i << (i == vars.u.rows() - 1 ? "" : ",");
    file << "\n";

    for (int k = 0; k <= N; ++k) {
      file << k << ",";
      for (int i = 0; i < vars.x.rows(); ++i)
        file << vars.x(i, k) << ",";
      if (k < N) {
        for (int i = 0; i < vars.u.rows(); ++i)
          file << vars.u(i, k) << (i == vars.u.rows() - 1 ? "" : ",");
      } else {
        for (int i = 0; i < vars.u.rows(); ++i)
          file << "0" << (i == vars.u.rows() - 1 ? "" : ",");
      }
      file << "\n";
    }
    file.close();
    std::cout << "Trajectory saved to " << filename << "\n";
  }
}
