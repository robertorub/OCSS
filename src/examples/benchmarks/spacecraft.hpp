#pragma once

#include "common.hpp"
#include <array>
#include <memory>
#include <ocss/admm.hpp>

using namespace ocss;

template <int N> void test_spacecraft() {
  print_test_banner("Spacecraft Rendezvous (HCW)");
  using scalar = double;
  constexpr int nx = 6;
  constexpr int nu = 3;
  // constexpr int N = 30; // Now template parameter

  scalar n = 0.0011, dt = 30.0, m = 500.0;
  Eigen::Matrix<scalar, nx, nx> Ac;
  Ac.setZero();
  Ac(0, 3) = 1;
  Ac(1, 4) = 1;
  Ac(2, 5) = 1;
  Ac(3, 0) = 3 * n * n;
  Ac(3, 4) = 2 * n;
  Ac(4, 3) = -2 * n;
  Ac(5, 2) = -n * n;

  Eigen::Matrix<scalar, nx, nu> Bc;
  Bc.setZero();
  Bc.block(3, 0, 3, 3) = (1.0 / m) * Eigen::Matrix<scalar, 3, 3>::Identity();

  Eigen::Matrix<scalar, nx, nx> Ad =
      Eigen::Matrix<scalar, nx, nx>::Identity() + Ac * dt;
  Eigen::Matrix<scalar, nx, nu> Bd = Bc * dt;

  Eigen::Matrix<scalar, nx, nx> Q;
  Q.setZero();
  Q.diagonal() << 1, 1, 1, 10, 10, 10;
  Eigen::Matrix<scalar, nu, nu> R = Eigen::Matrix<scalar, nu, nu>::Identity();
  Eigen::Matrix<scalar, nx, nx> Pn = Q * 10.0;

  BoxData<scalar, nu, 0> u_box;
  u_box.lb.setConstant(-1.0);
  u_box.ub.setConstant(1.0);

  // State constraints [-1000, 1000]
  BoxData<scalar, nx, 0> x_box;
  x_box.lb.setConstant(-1000.0);
  x_box.ub.setConstant(1000.0);

  StageConstraints<scalar, nx, nx, BoxData<scalar, nx, 0>> sc;
  sc.F.setIdentity();
  sc.FTF.setIdentity();
  sc.projectors = std::make_tuple(x_box);

  OptimalControlData<scalar, nx, nu, N, nx, true, true, BoxData<scalar, nu, 0>,
                     BoxData<scalar, nx, 0>>
      ocp;

  std::array<Eigen::Matrix<scalar, nx, nx>, N> A_arr;
  A_arr.fill(Ad);
  std::array<Eigen::Matrix<scalar, nx, nu>, N> B_arr;
  B_arr.fill(Bd);
  std::array<Eigen::Matrix<scalar, nx, nx>, N> Q_arr;
  Q_arr.fill(Q);
  std::array<Eigen::Matrix<scalar, nu, nu>, N> R_arr;
  R_arr.fill(R);

  ocp.set_Ad(A_arr);
  ocp.set_Bd(B_arr);
  ocp.set_Q(Q_arr);
  ocp.set_R(R_arr);
  ocp.set_Pn(Pn);
  ocp.set_q(Eigen::Matrix<scalar, nx, N>::Zero());
  ocp.set_pn(Eigen::Matrix<scalar, nx, 1>::Zero());
  ocp.set_b(Eigen::Matrix<scalar, nx, N>::Zero());
  ocp.set_s(Eigen::Matrix<scalar, nu, N>::Zero());
  ocp.set_input_constraints(u_box);
  ocp.set_state_constraints(sc);

  Options<scalar> opts;
  opts.rel_tol = 1e-4;
  opts.abs_tol = 1e-4;
  opts.max_iterations = 2000;
  opts.rho = 0.5; // Tune rho for faster convergence (scale with Q)

  auto solver_ptr =
      std::make_unique<ADMM<scalar, nx, nu, N, nx, nu, true, true,
                            BoxData<scalar, nu, 0>, BoxData<scalar, nx, 0>>>(
          ocp, opts);

  Eigen::Matrix<scalar, nx, 1> x0;
  x0.setZero();
  x0(0) = -100.0;
  x0(4) = 0.1;

  auto start = std::chrono::high_resolution_clock::now();
  const auto &res = solver_ptr->solve(x0);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  analyze_result(*solver_ptr, ocp, res, ms, "Spacecraft");
}
