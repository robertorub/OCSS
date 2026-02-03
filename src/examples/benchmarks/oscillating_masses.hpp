#pragma once

#include "common.hpp"
#include <array>
#include <memory>
#include <ocss/admm.hpp>

using namespace ocss;

template <int N> void test_oscillating_masses() {
  print_test_banner("Oscillating Masses (OSQP)");
  using scalar = double;
  constexpr int num_masses = 4;
  constexpr int nx = 2 * num_masses;
  constexpr int nu = num_masses;
  // constexpr int N = 30; // Now template parameter

  Eigen::Matrix<scalar, nx, nx> Ad;
  Eigen::Matrix<scalar, nx, nu> Bd;

  scalar m = 1.0, k_spring = 1.0, d_damp = 0.1, dt = 0.5;
  Eigen::Matrix<scalar, nx, nx> Ac;
  Ac.setZero();
  Ac.block(0, num_masses, num_masses, num_masses).setIdentity();

  Eigen::Matrix<scalar, num_masses, num_masses> K;
  K.setZero();
  for (int i = 0; i < num_masses; ++i) {
    K(i, i) = 2 * k_spring;
    if (i > 0)
      K(i, i - 1) = -k_spring;
    if (i < num_masses - 1)
      K(i, i + 1) = -k_spring;
  }
  Eigen::Matrix<scalar, num_masses, num_masses> D =
      d_damp * Eigen::Matrix<scalar, num_masses, num_masses>::Identity();

  Ac.block(num_masses, 0, num_masses, num_masses) = -(1.0 / m) * K;
  Ac.block(num_masses, num_masses, num_masses, num_masses) = -(1.0 / m) * D;

  Eigen::Matrix<scalar, nx, nu> Bc;
  Bc.setZero();
  Bc.block(num_masses, 0, num_masses, num_masses) =
      (1.0 / m) * Eigen::Matrix<scalar, num_masses, num_masses>::Identity();

  Ad = Eigen::Matrix<scalar, nx, nx>::Identity() + Ac * dt;
  Bd = Bc * dt;

  Eigen::Matrix<scalar, nx, nx> Q =
      2.0 * Eigen::Matrix<scalar, nx, nx>::Identity();
  Eigen::Matrix<scalar, nu, nu> R = Eigen::Matrix<scalar, nu, nu>::Identity();
  Eigen::Matrix<scalar, nx, nx> Pn = Q;

  BoxData<scalar, nu, 0> u_box;
  u_box.lb.setConstant(-1.0);
  u_box.ub.setConstant(1.0);

  BoxData<scalar, nx, 0> x_box;
  x_box.lb.setConstant(-1e10);
  x_box.ub.setConstant(1e10);
  x_box.lb.head(num_masses).setConstant(-4.0);
  x_box.ub.head(num_masses).setConstant(4.0);

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

  auto solver_ptr =
      std::make_unique<ADMM<scalar, nx, nu, N, nx, nu, true, true,
                            BoxData<scalar, nu, 0>, BoxData<scalar, nx, 0>>>(
          ocp, opts);

  Eigen::Matrix<scalar, nx, 1> x0;
  x0.setZero();
  for (int i = 0; i < num_masses; ++i)
    x0[i] = (i < num_masses / 2) ? 1.0 : -1.0;

  auto start = std::chrono::high_resolution_clock::now();
  const auto &res = solver_ptr->solve(x0);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  analyze_result(*solver_ptr, ocp, res, ms, "Oscillating_Masses");
}
