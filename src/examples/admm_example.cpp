#include <ocss/admm.hpp>

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <iostream>

using Clock = std::chrono::steady_clock;

template <typename scalar> void admm_example() {
  constexpr int N = 20;
  constexpr int nx = 6; // [px py pz vx vy vz]
  constexpr int nu = 3; // [ax ay az]
  const scalar dt = scalar(0.1);

  // --- Dynamics: 3D double integrator ---
  Eigen::Matrix<scalar, nx, nx> Ad = Eigen::Matrix<scalar, nx, nx>::Zero();
  Eigen::Matrix<scalar, nx, nu> Bd = Eigen::Matrix<scalar, nx, nu>::Zero();

  Ad.template block<3, 3>(0, 0).setIdentity();
  Ad.template block<3, 3>(0, 3) = dt * Eigen::Matrix<scalar, 3, 3>::Identity();
  Ad.template block<3, 3>(3, 3).setIdentity();

  Bd.template block<3, 3>(0, 0) =
      scalar(0.5) * dt * dt * Eigen::Matrix<scalar, 3, 3>::Identity();
  Bd.template block<3, 3>(3, 0) = dt * Eigen::Matrix<scalar, 3, 3>::Identity();

  // --- Cost ---
  Eigen::Matrix<scalar, nx, nx> Q = Eigen::Matrix<scalar, nx, nx>::Zero();
  Eigen::Matrix<scalar, nu, nu> R =
      scalar(0.05) * Eigen::Matrix<scalar, nu, nu>::Identity();

  Q.template block<3, 3>(0, 0) =
      scalar(10.0) * Eigen::Matrix<scalar, 3, 3>::Identity();
  Q.template block<3, 3>(3, 3) =
      scalar(1.0) * Eigen::Matrix<scalar, 3, 3>::Identity();
  Eigen::Matrix<scalar, nx, nx> Pn = Q;

  // Reference
  Eigen::Matrix<scalar, nx, 1> xr = Eigen::Matrix<scalar, nx, 1>::Zero();
  xr.template segment<3>(0) << scalar(1.5), scalar(-1.0), scalar(2.0);

  Eigen::Matrix<scalar, nx, 1> x0 = Eigen::Matrix<scalar, nx, 1>::Zero();
  x0.template segment<3>(0) << scalar(0.0), scalar(0.0), scalar(0.0);

  // affine terms (none)
  Eigen::Matrix<scalar, nx, 1> b = Eigen::Matrix<scalar, nx, 1>::Zero();

  // linear term q = -Q*xr, s = 0, pn = -Pn*xr
  Eigen::Matrix<scalar, nx, 1> q_stage = -Q * xr;
  Eigen::Matrix<scalar, nu, 1> s_stage = Eigen::Matrix<scalar, nu, 1>::Zero();
  Eigen::Matrix<scalar, nx, 1> pn = -Pn * xr;

  // --- Control bounds (box on acceleration) ---
  Eigen::Matrix<scalar, nu, 1> u_lb =
      Eigen::Matrix<scalar, nu, 1>::Constant(scalar(-2.0));
  Eigen::Matrix<scalar, nu, 1> u_ub =
      Eigen::Matrix<scalar, nu, 1>::Constant(scalar(2.0));

  // --- State constraints pattern: Box on p, Ball on v ---
  constexpr int d_box = 3;  // p
  constexpr int d_ball = 3; // v
  constexpr int z_dim = d_box + d_ball;

  Eigen::Matrix<scalar, z_dim, nx> F = Eigen::Matrix<scalar, z_dim, nx>::Zero();
  F.template block<3, 3>(0, 0).setIdentity();
  F.template block<3, 3>(3, 3).setIdentity();

  Eigen::Matrix<scalar, nx, nx> FTF = F.transpose() * F;

  // Projectors
  ocss::BoxData<scalar, d_box, 0> position_constraint;
  position_constraint.lb << scalar(-2.0), scalar(-2.0), scalar(0.0);
  position_constraint.ub << scalar(2.0), scalar(2.0), scalar(3.0);

  ocss::BallData<scalar, d_ball, d_box> velocity_constraint;
  velocity_constraint.radius = scalar(1.0);

  ocss::StageConstraints<scalar, nx, z_dim, ocss::BoxData<scalar, d_box, 0>,
                         ocss::BallData<scalar, d_ball, d_box>>
      sc;
  sc.F = F;
  sc.FTF = FTF;
  sc.projectors = std::make_tuple(position_constraint, velocity_constraint);

  ocss::BoxData<scalar, nu, 0> input_constraint;
  input_constraint.lb = u_lb;
  input_constraint.ub = u_ub;

  // --- Build horizon vectors (constant data repeated) ---
  std::array<Eigen::Matrix<scalar, nx, nx>, N> A_vec, Q_vec;
  std::array<Eigen::Matrix<scalar, nx, nu>, N> B_vec;
  std::array<Eigen::Matrix<scalar, nu, nu>, N> R_vec;
  Eigen::Matrix<scalar, nx, N> b_mat;
  Eigen::Matrix<scalar, nx, N> q_mat;
  Eigen::Matrix<scalar, nu, N> s_mat;

  for (int k = 0; k < N; ++k) {
    A_vec[k] = Ad;
    B_vec[k] = Bd;
    Q_vec[k] = Q;
    R_vec[k] = R;
    b_mat.col(k) = b;
    q_mat.col(k) = q_stage;
    s_mat.col(k) = s_stage;
  }

  ocss::Options<scalar> opt;
  opt.rho = scalar(1.0);
  opt.alpha = scalar(1.5);
  opt.max_iterations = 1000;
  opt.abs_tol = scalar(1e-4);
  opt.rel_tol = scalar(1e-3);

  ocss::OptimalControlData<
      scalar, nx, nu, N, z_dim, true, true, ocss::BoxData<scalar, nu, 0>,
      ocss::BoxData<scalar, d_box, 0>, ocss::BallData<scalar, d_ball, d_box>>
      ocp;

  ocp.set_Ad(A_vec);
  ocp.set_Bd(B_vec);
  ocp.set_Q(Q_vec);
  ocp.set_R(R_vec);
  ocp.set_Pn(Pn);
  ocp.set_b(b_mat);
  ocp.set_q(q_mat);
  ocp.set_s(s_mat);
  ocp.set_pn(pn);
  ocp.set_input_constraints(input_constraint);
  ocp.set_state_constraints(sc);

  ocss::ADMM<scalar, nx, nu, N, z_dim, nu, true, true,
             ocss::BoxData<scalar, nu, 0>, ocss::BoxData<scalar, d_box, 0>,
             ocss::BallData<scalar, d_ball, d_box>>
      solver(ocp, opt);

  const auto t0 = Clock::now();
  auto sol = solver.solve(x0);
  const auto t1 = Clock::now();

  const auto run_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  std::cout << "ADMM run time: " << run_time_us << " us\n";
  std::cout << "Optimal state trajectory:\n" << sol.x.col(1) << "\n";
  std::cout << "Optimal input trajectory:\n" << sol.u.col(0) << "\n";
}

/*
template <typename scalar> void admm_example_scaled() {
  constexpr int N = 10;
  constexpr int nx = 6; // [px py pz vx vy vz]
  constexpr int nu = 3; // [ax ay az]
  const scalar dt = scalar(0.1);

  // -------------------------
  // Choose scaling (tune these)
  // -------------------------
  // Typical magnitudes (example):
  // position ~ 2 m, velocity ~ 1 m/s, acceleration ~ 2 m/s^2
  Eigen::Matrix<scalar, nx, 1> sx;
  sx << scalar(2.0), scalar(2.0), scalar(2.0), scalar(1.0), scalar(1.0),
      scalar(1.0);

  Eigen::Matrix<scalar, nu, 1> su;
  su << scalar(2.0), scalar(2.0), scalar(2.0);

  const auto Sx = sx.asDiagonal();
  const auto Su = su.asDiagonal();
  const auto Sx_inv = sx.cwiseInverse().asDiagonal();
  const auto Su_inv = su.cwiseInverse().asDiagonal();

  // -------------------------
  // Original (physical) problem data
  // -------------------------
  Eigen::Matrix<scalar, nx, nx> A = Eigen::Matrix<scalar, nx, nx>::Zero();
  Eigen::Matrix<scalar, nx, nu> B = Eigen::Matrix<scalar, nx, nu>::Zero();

  A.template block<3, 3>(0, 0).setIdentity();
  A.template block<3, 3>(0, 3) = dt * Eigen::Matrix<scalar, 3, 3>::Identity();
  A.template block<3, 3>(3, 3).setIdentity();

  B.template block<3, 3>(0, 0) =
      scalar(0.5) * dt * dt * Eigen::Matrix<scalar, 3, 3>::Identity();
  B.template block<3, 3>(3, 0) = dt * Eigen::Matrix<scalar, 3, 3>::Identity();

  Eigen::Matrix<scalar, nx, nx> Q = Eigen::Matrix<scalar, nx, nx>::Zero();
  Eigen::Matrix<scalar, nu, nu> R =
      scalar(0.05) * Eigen::Matrix<scalar, nu, nu>::Identity();
  Q.template block<3, 3>(0, 0) =
      scalar(10.0) * Eigen::Matrix<scalar, 3, 3>::Identity();
  Q.template block<3, 3>(3, 3) =
      scalar(1.0) * Eigen::Matrix<scalar, 3, 3>::Identity();
  Eigen::Matrix<scalar, nx, nx> Pn = Q;

  Eigen::Matrix<scalar, nx, 1> xr = Eigen::Matrix<scalar, nx, 1>::Zero();
  xr.template segment<3>(0) << scalar(1.5), scalar(-1.0), scalar(2.0);

  Eigen::Matrix<scalar, nx, 1> x0 = Eigen::Matrix<scalar, nx, 1>::Zero();
  x0.template segment<3>(0) << scalar(0.0), scalar(0.0), scalar(0.0);

  Eigen::Matrix<scalar, nx, 1> b = Eigen::Matrix<scalar, nx, 1>::Zero();
  Eigen::Matrix<scalar, nx, 1> q_stage = -Q * xr;
  Eigen::Matrix<scalar, nu, 1> s_stage = Eigen::Matrix<scalar, nu, 1>::Zero();
  Eigen::Matrix<scalar, nx, 1> pn = -Pn * xr;

  Eigen::Matrix<scalar, nu, 1> u_lb =
      Eigen::Matrix<scalar, nu, 1>::Constant(scalar(-2.0));
  Eigen::Matrix<scalar, nu, 1> u_ub =
      Eigen::Matrix<scalar, nu, 1>::Constant(scalar(2.0));

  // State constraints z = F x, with z = [p; v]
  constexpr int d_box = 3;
  constexpr int d_ball = 3;
  constexpr int z_dim = d_box + d_ball;

  Eigen::Matrix<scalar, z_dim, nx> F = Eigen::Matrix<scalar, z_dim, nx>::Zero();
  F.template block<3, 3>(0, 0).setIdentity();
  F.template block<3, 3>(3, 3).setIdentity();

  // Projectors in PHYSICAL z-space (keep these unchanged)
  ocss::BoxData<scalar, d_box, 0> position_constraint;
  position_constraint.lb << scalar(-2.0), scalar(-2.0), scalar(0.0);
  position_constraint.ub << scalar(2.0), scalar(2.0), scalar(3.0);

  ocss::BallData<scalar, d_ball, d_box> velocity_constraint;
  velocity_constraint.radius = scalar(1.0);

  // -------------------------
  // Build SCALED problem data
  // -------------------------
  // Dynamics
  const Eigen::Matrix<scalar, nx, nx> Abar = Sx_inv * A * Sx;
  const Eigen::Matrix<scalar, nx, nu> Bbar = Sx_inv * B * Su;
  const Eigen::Matrix<scalar, nx, 1> bbar = Sx_inv * b;

  // Costs
  const Eigen::Matrix<scalar, nx, nx> Qbar = Sx * Q * Sx;
  const Eigen::Matrix<scalar, nu, nu> Rbar = Su * R * Su;
  const Eigen::Matrix<scalar, nx, nx> Pnbar = Sx * Pn * Sx;

  const Eigen::Matrix<scalar, nx, 1> qbar_stage = Sx * q_stage;
  const Eigen::Matrix<scalar, nu, 1> sbar_stage = Su * s_stage;
  const Eigen::Matrix<scalar, nx, 1> pnbar = Sx * pn;

  // Initial condition in scaled coordinates
  const Eigen::Matrix<scalar, nx, 1> x0_bar = Sx_inv * x0;

  // Input bounds become bounds on u_bar
  const Eigen::Matrix<scalar, nu, 1> u_lb_bar = Su_inv * u_lb;
  const Eigen::Matrix<scalar, nu, 1> u_ub_bar = Su_inv * u_ub;

  // IMPORTANT: Keep z physical by setting Fbar = F * Sx
  // Then (Fbar * x_bar) = (F * x) in physical units, so projector bounds/radius
  // stay unchanged.
  const Eigen::Matrix<scalar, z_dim, nx> Fbar = F * Sx;
  const Eigen::Matrix<scalar, nx, nx> FTFbar = Fbar.transpose() * Fbar;

  // Build constraint objects
  ocss::StageConstraints<scalar, nx, z_dim, ocss::BoxData<scalar, d_box, 0>,
                         ocss::BallData<scalar, d_ball, d_box>>
      sc;
  sc.F = Fbar;
  sc.FTF = FTFbar;
  sc.projectors = std::make_tuple(position_constraint, velocity_constraint);

  ocss::BoxData<scalar, nu, 0> input_constraint;
  input_constraint.lb = u_lb_bar;
  input_constraint.ub = u_ub_bar;

  // Horizon arrays
  std::array<Eigen::Matrix<scalar, nx, nx>, N> A_vec, Q_vec;
  std::array<Eigen::Matrix<scalar, nx, nu>, N> B_vec;
  std::array<Eigen::Matrix<scalar, nu, nu>, N> R_vec;
  Eigen::Matrix<scalar, nx, N> b_mat;
  Eigen::Matrix<scalar, nx, N> q_mat;
  Eigen::Matrix<scalar, nu, N> s_mat;

  for (int k = 0; k < N; ++k) {
    A_vec[k] = Abar;
    B_vec[k] = Bbar;
    Q_vec[k] = Qbar;
    R_vec[k] = Rbar;
    b_mat.col(k) = bbar;
    q_mat.col(k) = qbar_stage;
    s_mat.col(k) = sbar_stage;
  }

  ocss::Options<scalar> opt;
  opt.rho = scalar(1.0); // after scaling, rho ~ O(1) is usually reasonable
  opt.alpha = scalar(1.5);
  opt.max_iterations = 1000;
  opt.abs_tol = scalar(1e-4);
  opt.rel_tol = scalar(1e-3);

  // Build scaled OCP
  ocss::OptimalControlData<
      scalar, nx, nu, N, z_dim, true, true, ocss::BoxData<scalar, nu, 0>,
      ocss::BoxData<scalar, d_box, 0>, ocss::BallData<scalar, d_ball, d_box>>
      ocp_bar{x0_bar,
              A_vec,
              B_vec,
              Q_vec,
              R_vec,
              Pnbar,
              b_mat,
              q_mat,
              s_mat,
              pnbar,
              input_constraint,
              sc};

  ocss::ADMM<scalar, nx, nu, N, z_dim, nu, ocss::BoxData<scalar, nu, 0>,
             ocss::BoxData<scalar, d_box, 0>,
             ocss::BallData<scalar, d_ball, d_box>>
      solver(ocp_bar, opt);

  // Solve in scaled coordinates
  const auto t0 = Clock::now();
  auto sol_bar = solver.solve(x0_bar);
  const auto t1 = Clock::now();

  const auto run_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "ADMM run time: " << run_time_us << " us\n";

  // Map solution back to physical units:
  // x = Sx * x_bar, u = Su * u_bar
  Eigen::Matrix<scalar, nx, N + 1> x_phys;
  Eigen::Matrix<scalar, nu, N> u_phys;
  x_phys.noalias() = Sx * sol_bar.x;
  u_phys.noalias() = Su * sol_bar.u;

  std::cout << "Optimal state trajectory (physical):\n" << x_phys << "\n";
  std::cout << "Optimal input trajectory (physical):\n" << u_phys << "\n";

  // -------------------------
  // Constraint checks (PHYSICAL units)
  // -------------------------
  const scalar tol = scalar(1e-3);

  // Rebuild physical constraint data for checks
  const Eigen::Matrix<scalar, 3, 1> p_lb =
      (Eigen::Matrix<scalar, 3, 1>() << scalar(-2.0), scalar(-2.0), scalar(0.0))
          .finished();
  const Eigen::Matrix<scalar, 3, 1> p_ub =
      (Eigen::Matrix<scalar, 3, 1>() << scalar(2.0), scalar(2.0), scalar(3.0))
          .finished();
  const scalar v_radius = scalar(1.0);

  // Track maximum violations (positive => violated)
  scalar max_u_viol = scalar(0);
  scalar max_p_viol = scalar(0);
  scalar max_v_viol = scalar(0);

  int k_u_max = -1, k_p_max = -1, k_v_max = -1;

  // 1) Input bounds
  for (int k = 0; k < N; ++k) {
    const Eigen::Matrix<scalar, nu, 1> uk = u_phys.col(k);

    const Eigen::Matrix<scalar, nu, 1> viol_low =
        (u_lb - uk).cwiseMax(scalar(0));
    const Eigen::Matrix<scalar, nu, 1> viol_high =
        (uk - u_ub).cwiseMax(scalar(0));
    const scalar viol = std::max(viol_low.maxCoeff(), viol_high.maxCoeff());

    if (viol > max_u_viol) {
      max_u_viol = viol;
      k_u_max = k;
    }
  }

  // 2) Position box bounds, 3) Velocity ball
  for (int k = 0; k <= N; ++k) {
    const Eigen::Matrix<scalar, 3, 1> pk = x_phys.template block<3, 1>(0, k);
    const Eigen::Matrix<scalar, 3, 1> vk = x_phys.template block<3, 1>(3, k);

    // position box violation
    const Eigen::Matrix<scalar, 3, 1> p_viol_low =
        (p_lb - pk).cwiseMax(scalar(0));
    const Eigen::Matrix<scalar, 3, 1> p_viol_high =
        (pk - p_ub).cwiseMax(scalar(0));
    const scalar p_viol =
        std::max(p_viol_low.maxCoeff(), p_viol_high.maxCoeff());
    if (p_viol > max_p_viol) {
      max_p_viol = p_viol;
      k_p_max = k;
    }

    // velocity ball violation: ||v|| - r
    const scalar v_norm = vk.norm();
    const scalar v_viol = std::max(v_norm - v_radius, scalar(0));
    if (v_viol > max_v_viol) {
      max_v_viol = v_viol;
      k_v_max = k;
    }
  }

  std::cout << "\n--- Constraint check (physical units) ---\n";
  std::cout << "max input box violation  : " << max_u_viol
            << "  at k=" << k_u_max << "\n";
  std::cout << "max position box violation: " << max_p_viol
            << "  at k=" << k_p_max << "\n";
  std::cout << "max velocity ball violation: " << max_v_viol
            << "  at k=" << k_v_max << "\n";

  const bool ok_u = (max_u_viol <= tol);
  const bool ok_p = (max_p_viol <= tol);
  const bool ok_v = (max_v_viol <= tol);

  std::cout << "input bounds satisfied   : " << (ok_u ? "YES" : "NO") << "\n";
  std::cout << "position bounds satisfied: " << (ok_p ? "YES" : "NO") << "\n";
  std::cout << "velocity ball satisfied  : " << (ok_v ? "YES" : "NO") << "\n";

  // Note: sol_bar.x_tilde is in PHYSICAL z-space already (because Fbar = F*Sx),
  // so you can interpret x_tilde directly as [p; v] in meters and m/s.
}

template <typename scalar> void admm_qp_example() {
  constexpr int N = 20;
  constexpr int nx = 6; // [px py pz vx vy vz]
  constexpr int nu = 3; // [ax ay az]
  const scalar dt = scalar(0.1);

  // --- Dynamics: 3D double integrator ---
  Eigen::Matrix<scalar, nx, nx> Ad = Eigen::Matrix<scalar, nx, nx>::Zero();
  Eigen::Matrix<scalar, nx, nu> Bd = Eigen::Matrix<scalar, nx, nu>::Zero();

  Ad.template block<3, 3>(0, 0).setIdentity();
  Ad.template block<3, 3>(0, 3) = dt * Eigen::Matrix<scalar, 3, 3>::Identity();
  Ad.template block<3, 3>(3, 3).setIdentity();

  Bd.template block<3, 3>(0, 0) =
      scalar(0.5) * dt * dt * Eigen::Matrix<scalar, 3, 3>::Identity();
  Bd.template block<3, 3>(3, 0) = dt * Eigen::Matrix<scalar, 3, 3>::Identity();

  // --- Cost ---
  Eigen::Matrix<scalar, nx, nx> Q = Eigen::Matrix<scalar, nx, nx>::Zero();
  Eigen::Matrix<scalar, nu, nu> R =
      scalar(0.05) * Eigen::Matrix<scalar, nu, nu>::Identity();

  Q.template block<3, 3>(0, 0) =
      scalar(10.0) * Eigen::Matrix<scalar, 3, 3>::Identity();
  Q.template block<3, 3>(3, 3) =
      scalar(1.0) * Eigen::Matrix<scalar, 3, 3>::Identity();
  Eigen::Matrix<scalar, nx, nx> Pn = Q;

  // Reference
  Eigen::Matrix<scalar, nx, 1> xr = Eigen::Matrix<scalar, nx, 1>::Zero();
  xr.template segment<3>(0) << scalar(1.5), scalar(-1.0), scalar(2.0);

  Eigen::Matrix<scalar, nx, 1> x0 = Eigen::Matrix<scalar, nx, 1>::Zero();
  x0.template segment<3>(0) << scalar(0.0), scalar(0.0), scalar(0.0);

  // affine terms (none)
  Eigen::Matrix<scalar, nx, 1> b = Eigen::Matrix<scalar, nx, 1>::Zero();

  // linear term q = -Q*xr, s = 0, pn = -Pn*xr
  Eigen::Matrix<scalar, nx, 1> q_stage = -Q * xr;
  Eigen::Matrix<scalar, nu, 1> s_stage = Eigen::Matrix<scalar, nu, 1>::Zero();
  Eigen::Matrix<scalar, nx, 1> pn = -Pn * xr;

  // --- Control bounds (box on acceleration) ---
  Eigen::Matrix<scalar, nu, 1> u_lb =
      Eigen::Matrix<scalar, nu, 1>::Constant(scalar(-2.0));
  Eigen::Matrix<scalar, nu, 1> u_ub =
      Eigen::Matrix<scalar, nu, 1>::Constant(scalar(2.0));

  // --- State constraints pattern: Box on p, Ball on v ---
  constexpr int d_box_p = 3; // p
  constexpr int d_box_v = 3; // v
  constexpr int z_dim = d_box_p + d_box_v;

  Eigen::Matrix<scalar, z_dim, nx> F = Eigen::Matrix<scalar, z_dim, nx>::Zero();
  F.template block<3, 3>(0, 0).setIdentity();
  F.template block<3, 3>(3, 3).setIdentity();

  Eigen::Matrix<scalar, nx, nx> FTF = F.transpose() * F;

  // Projectors
  ocss::BoxData<scalar, d_box_p, 0> position_constraint;
  position_constraint.lb << scalar(-2.0), scalar(-2.0), scalar(0.0);
  position_constraint.ub << scalar(2.0), scalar(2.0), scalar(3.0);

  ocss::BoxData<scalar, d_box_v, d_box_p> velocity_constraint;
  velocity_constraint.lb << scalar(-1.0), scalar(-1.0), scalar(-1.0);
  velocity_constraint.ub << scalar(1.0), scalar(1.0), scalar(1.0);

  ocss::StageConstraints<scalar, nx, z_dim, ocss::BoxData<scalar, d_box_p, 0>,
                         ocss::BoxData<scalar, d_box_v, d_box_p>>
      sc;
  sc.F = F;
  sc.FTF = FTF;
  sc.projectors = std::make_tuple(position_constraint, velocity_constraint);

  ocss::BoxData<scalar, nu, 0> input_constraint;
  input_constraint.lb = u_lb;
  input_constraint.ub = u_ub;

  // --- Build horizon vectors (constant data repeated) ---
  std::array<Eigen::Matrix<scalar, nx, nx>, N> A_vec, Q_vec;
  std::array<Eigen::Matrix<scalar, nx, nu>, N> B_vec;
  std::array<Eigen::Matrix<scalar, nu, nu>, N> R_vec;
  Eigen::Matrix<scalar, nx, N> b_mat;
  Eigen::Matrix<scalar, nx, N> q_mat;
  Eigen::Matrix<scalar, nu, N> s_mat;

  for (int k = 0; k < N; ++k) {
    A_vec[k] = Ad;
    B_vec[k] = Bd;
    Q_vec[k] = Q;
    R_vec[k] = R;
    b_mat.col(k) = b;
    q_mat.col(k) = q_stage;
    s_mat.col(k) = s_stage;
  }

  ocss::Options<scalar> opt;
  opt.rho = scalar(1.0);
  opt.alpha = scalar(1.6);
  opt.max_iterations = 1000;
  opt.abs_tol = scalar(1e-4);
  opt.rel_tol = scalar(1e-3);

  ocss::OptimalControlData<scalar, nx, nu, N, z_dim,
                           true, true, ocss::BoxData<scalar, nu, 0>,
                           ocss::BoxData<scalar, d_box_p, 0>,
                           ocss::BoxData<scalar, d_box_v, d_box_p>>
      ocp;

  ocp.set_Ad(A_vec);
  ocp.set_Bd(B_vec);
  ocp.set_Q(Q_vec);
  ocp.set_R(R_vec);
  ocp.set_Pn(Pn);
  ocp.set_b(b_mat);
  ocp.set_q(q_mat);
  ocp.set_s(s_mat);
  ocp.set_pn(pn);
  ocp.set_input_constraints(input_constraint);
  ocp.set_state_constraints(sc);

  ocss::ADMM<scalar, nx, nu, N, z_dim, nu, ocss::BoxData<scalar, nu, 0>,
             ocss::BoxData<scalar, d_box_p, 0>,
             ocss::BoxData<scalar, d_box_v, d_box_p>>
      solver(ocp, opt);

  const auto t0 = Clock::now();
  auto sol = solver.solve(x0);
  const auto t1 = Clock::now();

  const auto run_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  std::cout << "ADMM run time: " << run_time_us << " us\n";
  std::cout << "Optimal state trajectory:\n" << sol.x << "\n";
  std::cout << "Optimal input trajectory:\n" << sol.u << "\n";
  std::cout << "x - x_tilde " << (sol.x - sol.x_tilde) << "\n";
}
*/
int main() {

  admm_example<double>();

  return 0;
}