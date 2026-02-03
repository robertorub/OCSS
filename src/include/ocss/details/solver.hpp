#pragma once
#include <iostream>

namespace ocss {

template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
void ADMM<scalar, nx, nu, N, nx_tilde, nu_tilde,
          UniformTrajectoryStateConstraints, UniformTrajectoryInputConstraints,
          InputSet, Projectors...>::update_q_and_s() {
  for (int i = 0; i < N; i++) {
    ws.tmp1_x_tilde = ws.variables.y_x.col(i);
    ws.tmp1_x_tilde.noalias() -= rho * ws.variables.x_tilde.col(i);
    ws.q_admm.col(i).noalias() =
        ocp.state_constraints[i].F.transpose() *
        ws.tmp1_x_tilde; // ocp.state_constraints.apply_FT(ws.q_admm.col(i),
                         // ws.tmp1_x_tilde);//ws.q_admm.col(i).noalias() =
                         // ocp.state_constraints.F.transpose()*ws.tmp1_x_tilde;
    ws.q_admm.col(i) += ocp.q.col(i);
    ws.s_admm.col(i) = ocp.s.col(i) + ws.variables.y_u.col(i);
    ws.s_admm.col(i).noalias() -= rho * ws.variables.u_tilde.col(i);
  }
  ws.tmp1_x_tilde = ws.variables.y_x.col(N);
  ws.tmp1_x_tilde.noalias() -= rho * ws.variables.x_tilde.col(N);
  ws.pn_admm.noalias() =
      ocp.state_constraints[N].F.transpose() *
      ws.tmp1_x_tilde; // ocp.state_constraints.apply_FT(ws.pn_admm,
                       // ws.tmp1_x_tilde);//ws.pn_admm.noalias() =
                       // ocp.state_constraints.F.transpose()*ws.tmp1_x_tilde;
  ws.pn_admm += ocp.pn;
}

template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
void ADMM<scalar, nx, nu, N, nx_tilde, nu_tilde,
          UniformTrajectoryStateConstraints, UniformTrajectoryInputConstraints,
          InputSet, Projectors...>::update_Q_and_R() {
  for (int i = 0; i < N; i++) {
    ws.Q_admm[i] = ocp.Q[i];
    ws.Q_admm[i].noalias() += rho * ocp.state_constraints[i].FTF;
    ws.R_admm[i] = ocp.R[i];
    ws.R_admm[i].diagonal().array() += rho;
  }
  ws.Pn_admm = ocp.Pn;
  ws.Pn_admm.noalias() += rho * ocp.state_constraints[N].FTF;
}

template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
void ADMM<scalar, nx, nu, N, nx_tilde, nu_tilde,
          UniformTrajectoryStateConstraints, UniformTrajectoryInputConstraints,
          InputSet, Projectors...>::step() {

  // (x,u) step

  riccati_solve();

  // x_tilde, u_tilde step and dual updates
  for (int i = 0; i < N; i++) {
    ws.variables.x_tilde_prev.col(i + 1) =
        ws.variables.x_tilde.col(i + 1); // i+1 because x[0] is x0
    ws.variables.u_tilde_prev.col(i) = ws.variables.u_tilde.col(i);
    // relaxation step
    ws.tmp1_u_tilde.noalias() = options.alpha * ws.variables.u.col(i);
    ws.tmp1_u_tilde.noalias() +=
        (1 - options.alpha) * ws.variables.u_tilde.col(i);
    // projection step
    ws.tmp2_u_tilde.noalias() = ws.tmp1_u_tilde;

    ws.tmp2_u_tilde.noalias() += (1 / rho) * ws.variables.y_u.col(i);
    auto u_tilde_i = ws.variables.u_tilde.col(i);

    ocp.input_constraints[i].project(u_tilde_i, ws.tmp2_u_tilde);

    // bound_projection(ws.variables.u_tilde.col(i), ws.tmp2_u_tilde,ocp.u_lb,
    // ocp.u_ub);
    //  dual update
    ws.variables.y_u.col(i).noalias() += rho * (ws.tmp1_u_tilde - u_tilde_i);

    // relaxation step
    ws.tmp2_x_tilde.noalias() =
        ocp.state_constraints[i].F *
        ws.variables.x.col(
            i + 1); // ocp.state_constraints.apply_F(ws.tmp2_x_tilde,
                    // ws.variables.x.col(i+1));//ws.tmp2_x_tilde.noalias() =
                    // ocp.state_constraints.F*ws.variables.x.col(i+1); // i+1
                    // because x[0] is x0

    ws.tmp1_x_tilde.noalias() =
        (1 - options.alpha) *
        ws.variables.x_tilde.col(i + 1); // i+1 because x[0] is x0
    ws.tmp1_x_tilde.noalias() += options.alpha * ws.tmp2_x_tilde;
    // projection step
    ws.tmp2_x_tilde.noalias() = ws.tmp1_x_tilde;
    ws.tmp2_x_tilde.noalias() +=
        (1 / rho) * ws.variables.y_x.col(i + 1); // i+1 because x[0] is x0
    // projections
    auto x_tilde_iplus1 = ws.variables.x_tilde.col(i + 1);
    ocp.state_constraints[i].apply_projections(x_tilde_iplus1, ws.tmp2_x_tilde);

    // dual update
    ws.variables.y_x.col(i + 1).noalias() +=
        rho * (ws.tmp1_x_tilde - x_tilde_iplus1); // i+1 because x[0] is x0
  }
};

template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
bool ADMM<scalar, nx, nu, N, nx_tilde, nu_tilde,
          UniformTrajectoryStateConstraints, UniformTrajectoryInputConstraints,
          InputSet, Projectors...>::converged() {
  // primal residual r_i = (Fx_i - x_tilde_i, u_i - u_tilde_i )
  // dual residual s_i = rho ( F'(x_tilde_i - x_tilde_i_prev), (u_tilde_i -
  // u_tilde_i_prev) ) norm(r) < sqrt(N*(nx+nu))eps_abs + eps_rel max(
  // norm(Fx,u), norm(x_tilde,u_tilde) ) norm(s) <
  // sqrt(N*(nx_tilde+nu_tilde))eps_abs + eps_rel norm( (F' y_x, y_u) )
  scalar r2 = 0.0;
  scalar s2 = 0.0;
  scalar r_bound_1 = 0.0;
  scalar r_bound_2 = 0.0;
  scalar s_bound = 0.0;

  for (int k = 0; k < ocp.N; k++) {
    s2 += (ws.variables.u_tilde.col(k) - ws.variables.u_tilde_prev.col(k))
              .squaredNorm();
    ws.tmp1_x_tilde.noalias() =
        ws.variables.x_tilde.col(k + 1) -
        ws.variables.x_tilde_prev.col(k + 1); // k+1 because x[0] is x0
    ws.tmp_x.noalias() =
        ocp.state_constraints[k].F.transpose() * ws.tmp1_x_tilde;
    s2 += ws.tmp_x.squaredNorm();
    ws.tmp_x.noalias() =
        ocp.state_constraints[k].F.transpose() * ws.variables.y_x.col(k + 1);

    s_bound += ws.tmp_x.squaredNorm();

    s_bound += ws.variables.y_u.col(k).squaredNorm();

    r2 += (ws.variables.u.col(k) - ws.variables.u_tilde.col(k)).squaredNorm();
    ws.tmp1_x_tilde.noalias() =
        ocp.state_constraints[k].F *
        ws.variables.x.col(
            k + 1); // ocp.state_constraints.apply_F(ws.tmp1_x_tilde,
                    // ws.variables.x.col(k+1)); //   ws.tmp1_x_tilde.noalias()
                    // = ocp.state_constraints.F*ws.variables.x.col(k+1); // k+1
                    // because x[0] is x0
    r_bound_1 += ws.tmp1_x_tilde.squaredNorm();
    ws.tmp2_x_tilde.noalias() =
        ws.tmp1_x_tilde -
        ws.variables.x_tilde.col(k + 1); // k+1 because x[0] is x0
    r2 += ws.tmp2_x_tilde.squaredNorm();
    r_bound_1 += ws.variables.u.col(k).squaredNorm();
    r_bound_2 += ws.variables.u_tilde.col(k).squaredNorm();
    r_bound_2 +=
        ws.variables.x_tilde.col(k + 1).squaredNorm(); // k+1 because x[0] is x0
  }

  scalar r_norm = std::sqrt(r2);
  scalar s_norm = std::sqrt(s2);
  s_norm = s_norm * rho;

  r_bound_1 = std::max(r_bound_1, r_bound_2);
  r_bound_1 = std::sqrt(r_bound_1);
  s_bound = std::sqrt(s_bound);
  const scalar scaling = std::sqrt(N * (nx + nu));
  if (r_norm < options.abs_tol * scaling + options.rel_tol * r_bound_1 &&
      s_norm < options.abs_tol * scaling + options.rel_tol * s_bound) {
    return true;
  }

  rho_scaling = std::sqrt(r_norm / s_norm);

  return false;
}

template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
const OptimizationVariables<scalar, nx, nu, N, nx_tilde, nu_tilde> &
ADMM<scalar, nx, nu, N, nx_tilde, nu_tilde, UniformTrajectoryStateConstraints,
     UniformTrajectoryInputConstraints, InputSet,
     Projectors...>::solve(const Eigen::Matrix<scalar, nx, 1> &_x0) {

  ws.variables.x.col(0) = _x0;
  update_Q_and_R();

  update_q_and_s();
  // Eigen::internal::set_is_malloc_allowed(false);
  riccati_factorization();

  // Eigen::internal::set_is_malloc_allowed(true);

  for (int i = 1; i < options.max_iterations; ++i) {

    step();

    if (i % 25 == 0) {

      if (converged()) {
        std::cout << "ADMM converged in " << i << " iterations.\n";

        return ws.variables;
      }

      // if (i == 25) {
      rho = std::max(options.rho_min, rho * rho_scaling);
      update_Q_and_R();
      riccati_factorization();
      //}
    }

    update_q_and_s();
  }
  return ws.variables;
}

} // namespace ocss