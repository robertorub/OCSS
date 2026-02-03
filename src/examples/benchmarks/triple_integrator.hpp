#pragma once

#include "common.hpp"
#include <array>
#include <memory>
#include <ocss/admm.hpp>

using namespace ocss;

template <int N> void test_triple_integrator() {
  print_test_banner("Triple Integrator (3D)");
  using scalar = double;
  constexpr int nx = 9; // p, v, a (3 each)
  constexpr int nu = 3; // jerk
  // constexpr int N = 20; // Now template parameter

  scalar dt = 0.1;

  // Dynamics
  // p+ = p + v*dt + 0.5*a*dt^2 + 1/6*j*dt^3
  // v+ = v + a*dt + 0.5*j*dt^2
  // a+ = a + j*dt

  Eigen::Matrix<scalar, nx, nx> Ad = Eigen::Matrix<scalar, nx, nx>::Identity();
  // p->v
  Ad.block<3, 3>(0, 3) = Eigen::Matrix<scalar, 3, 3>::Identity() * dt;
  // p->a
  Ad.block<3, 3>(0, 6) =
      Eigen::Matrix<scalar, 3, 3>::Identity() * 0.5 * dt * dt;
  // v->a
  Ad.block<3, 3>(3, 6) = Eigen::Matrix<scalar, 3, 3>::Identity() * dt;

  Eigen::Matrix<scalar, nx, nu> Bd;
  Bd.setZero();
  // p<-j
  Bd.block<3, 3>(0, 0) =
      Eigen::Matrix<scalar, 3, 3>::Identity() * (1.0 / 6.0) * dt * dt * dt;
  // v<-j
  Bd.block<3, 3>(3, 0) =
      Eigen::Matrix<scalar, 3, 3>::Identity() * 0.5 * dt * dt;
  // a<-j
  Bd.block<3, 3>(6, 0) = Eigen::Matrix<scalar, 3, 3>::Identity() * dt;

  Eigen::Matrix<scalar, nx, nx> Q = Eigen::Matrix<scalar, nx, nx>::Identity();
  Eigen::Matrix<scalar, nu, nu> R =
      0.1 * Eigen::Matrix<scalar, nu, nu>::Identity();
  Eigen::Matrix<scalar, nx, nx> Pn = Q;

  // Constraints
  // ||v|| <= 5.0
  // ||a|| <= 10.0
  // |j| <= 20.0

  // Input Box
  scalar j_max = 20.0;
  BoxData<scalar, nu, 0> u_box;
  u_box.lb.setConstant(-j_max);
  u_box.ub.setConstant(j_max);

  // State Constraints
  // We need to constrain norm(v) and norm(a).
  // v is indices 3,4,5. a is indices 6,7,8.
  // StageConstraints F * x
  // Let z = [v; a] (size 6)
  // F = [0 I 0; 0 0 I] (6x9)

  // Projectors:
  // 1. Ball on first 3 elements of z (v)
  // 2. Ball on next 3 elements of z (a)

  BallData<scalar, 3, 0> v_ball;
  v_ball.radius = 5.0;

  BallData<scalar, 3, 3> a_ball;
  a_ball.radius = 10.0;

  constexpr int nx_tilde = 6;
  StageConstraints<scalar, nx, nx_tilde, BallData<scalar, 3, 0>,
                   BallData<scalar, 3, 3>>
      sc;

  sc.F.setZero();
  // Map v (indices 3,4,5) to first 3 rows of z
  sc.F.block<3, 3>(0, 3).setIdentity();
  // Map a (indices 6,7,8) to next 3 rows of z
  sc.F.block<3, 3>(3, 6).setIdentity();

  // Define efficient FTF = F.T * F
  // F.T = [0 0; I 0; 0 I]
  // F.T * F = [0 0 0; 0 I 0; 0 0 I] (Diagonal with 0s for p and Is for v, a)
  sc.FTF.setZero();
  sc.FTF.block<3, 3>(3, 3).setIdentity();
  sc.FTF.block<3, 3>(6, 6).setIdentity();

  sc.projectors = std::make_tuple(v_ball, a_ball);

  OptimalControlData<scalar, nx, nu, N, nx_tilde, true, true,
                     BoxData<scalar, nu, 0>, BallData<scalar, 3, 0>,
                     BallData<scalar, 3, 3>>
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

  // Template parameters for ADMM: matches OCP
  auto solver_ptr = std::make_unique<
      ADMM<scalar, nx, nu, N, nx_tilde, nu, true, true, BoxData<scalar, nu, 0>,
           BallData<scalar, 3, 0>, BallData<scalar, 3, 3>>>(ocp, opts);

  Eigen::Matrix<scalar, nx, 1> x0;
  x0.setZero();
  x0.head<3>().setConstant(-2.0);

  auto start = std::chrono::high_resolution_clock::now();
  const auto &res = solver_ptr->solve(x0);
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  analyze_result(*solver_ptr, ocp, res, ms, "Triple_Integrator");
}
