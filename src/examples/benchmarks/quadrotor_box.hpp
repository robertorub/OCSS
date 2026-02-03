#pragma once

#include "common.hpp"
#include <array>
#include <memory>
#include <ocss/admm.hpp>

using namespace ocss;

template <int N> void test_linear_quadrotor() {
  print_test_banner("Linear Quadrotor (Box)");
  using scalar = double;
  constexpr int nx = 12;
  constexpr int nu = 4;
  // constexpr int N = 30; // Now template parameter

  Eigen::Matrix<scalar, nx, nx> Ad;
  Ad << 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
      0., 0.1, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.,
      0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0., 0., -0.0488,
      0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0., 0., 0., 0., 0., 0., 1.,
      0., 0., 0., 0., 0., 0.0992, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0., 0., 0., 1., 0., 0., 0., 0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0.,
      0.9846, 0., 0., 0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846,
      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846;

  Eigen::Matrix<scalar, nx, nu> Bd;
  Bd << 0., -0.0726, 0., 0.0726, -0.0726, 0., 0.0726, 0., -0.0152, 0.0152,
      -0.0152, 0.0152, -0., -0.0006, -0., 0.0006, 0.0006, 0., -0.0006, 0.0000,
      0.0106, 0.0106, 0.0106, 0.0106, 0, -1.4512, 0., 1.4512, -1.4512, 0.,
      1.4512, 0., -0.3049, 0.3049, -0.3049, 0.3049, -0., -0.0236, 0., 0.0236,
      0.0236, 0., -0.0236, 0., 0.2107, 0.2107, 0.2107, 0.2107;

  Eigen::Matrix<scalar, nx, nx> Q;
  Q.setZero();
  Q.diagonal() << 0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.;
  Eigen::Matrix<scalar, nu, nu> R =
      0.1 * Eigen::Matrix<scalar, nu, nu>::Identity();

  Eigen::Matrix<scalar, nx, 1> q_vec;
  q_vec.setZero();
  q_vec[2] = -Q(2, 2) * 1.0;

  scalar u0 = 10.5916;
  Eigen::Matrix<scalar, nu, 1> umin, umax;
  umin.setConstant(9.6 - u0);
  umax.setConstant(13.0 - u0);

  BoxData<scalar, nu, 0> u_box;
  u_box.lb = umin;
  u_box.ub = umax;

  BoxData<scalar, nx, 0> x_box;
  x_box.lb.setConstant(-1e10);
  x_box.ub.setConstant(1e10);
  scalar pi = 3.1415926535;
  x_box.lb[0] = -pi / 6.0;
  x_box.lb[1] = -pi / 6.0;
  x_box.lb[5] = -1.0;
  x_box.ub[0] = pi / 6.0;
  x_box.ub[1] = pi / 6.0;

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
  ocp.set_Pn(Q);
  Eigen::Matrix<scalar, nx, N> q_mat;
  for (int i = 0; i < N; ++i)
    q_mat.col(i) = q_vec;
  ocp.set_q(q_mat);
  ocp.set_pn(q_vec);
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

  double total_ms = 0.0;
  double max_ms = 0.0;

  // Warmup run (optional, but good practice, or just loop 10 times)
  auto start = std::chrono::high_resolution_clock::now();
  const auto &res = solver_ptr->solve(x0);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  analyze_result(*solver_ptr, ocp, res, ms, "Linear_Quadrotor");
}
