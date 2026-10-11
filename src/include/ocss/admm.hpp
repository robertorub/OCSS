#pragma once
#include "ocp.hpp"
#include "options.hpp"
#include "types.hpp"
#include <Eigen/Dense>
namespace ocss {

template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
class ADMM {
public:
  ADMM(const OptimalControlData<
           scalar, nx, nu, N, nx_tilde, UniformTrajectoryStateConstraints,
           UniformTrajectoryInputConstraints, InputSet, Projectors...> &ocp_,
       Options<scalar> options_)
      : ocp(ocp_), options(options_), ws(), status() {}

  const OptimizationVariables<scalar, nx, nu, N, nx_tilde, nu_tilde> &
  solve(const Eigen::Matrix<scalar, nx, 1> &_x0);
  void reset();
  const Status<scalar> &get_status() const { return status; }

  // largest violation of the hard sets by (x_tilde, u_tilde), soft sets only
  scalar soft_violation() const {
    scalar v = scalar(0);
    for (int k = 0; k < N; ++k) {
      v = std::max(v, ocp.state_constraints[k].violation(
                          ws.variables.x_tilde.col(k + 1)));
      v = std::max(v, violation_of(ocp.input_constraints[k],
                                   ws.variables.u_tilde.col(k)));
    }
    return v;
  }
  const Eigen::Matrix<scalar, nx_tilde, N + 1> &state_duals() const {
    return ws.variables.y_x;
  }
  const Eigen::Matrix<scalar, nu_tilde, N> &input_duals() const {
    return ws.variables.y_u;
  }

private:
  const OptimalControlData<
      scalar, nx, nu, N, nx_tilde, UniformTrajectoryStateConstraints,
      UniformTrajectoryInputConstraints, InputSet, Projectors...> &ocp;
  Options<scalar> options;
  ADMMWorkspace<scalar, nx, nu, N, nx_tilde, nu_tilde> ws;
  Status<scalar> status;
  scalar rho_scaling = scalar(1.0);
  scalar rho = options.rho;

  // F with at most one nonzero per row and column: F x and F' v as index operations
  bool f_sparse = false;
  Eigen::Matrix<int, nx_tilde, 1> f_col;
  Eigen::Matrix<scalar, nx_tilde, 1> f_val;
  void analyze_F() {
    f_sparse = false;
    if constexpr (!UniformTrajectoryStateConstraints)
      return;
    const auto &F = ocp.state_constraints[0].F;
    Eigen::Matrix<int, nx, 1> used = Eigen::Matrix<int, nx, 1>::Zero();
    for (int r = 0; r < nx_tilde; ++r) {
      f_col(r) = -1;
      f_val(r) = scalar(0);
      for (int c = 0; c < nx; ++c) {
        if (F(r, c) == scalar(0))
          continue;
        if (f_col(r) >= 0 || used(c))
          return;
        f_col(r) = c;
        f_val(r) = F(r, c);
        used(c) = 1;
      }
    }
    f_sparse = true;
  }
  template <typename DY, typename DX>
  void F_times(DY &&y, const DX &x, int k) const {
    if (f_sparse) {
      for (int r = 0; r < nx_tilde; ++r)
        y(r) = f_col(r) >= 0 ? f_val(r) * x(f_col(r)) : scalar(0);
    } else {
      y.noalias() = ocp.state_constraints[k].F * x;
    }
  }
  template <typename DY, typename DV>
  void FT_times(DY &&y, const DV &v, int k) const {
    if (f_sparse) {
      y.setZero();
      for (int r = 0; r < nx_tilde; ++r)
        if (f_col(r) >= 0)
          y(f_col(r)) = f_val(r) * v(r);
    } else {
      y.noalias() = ocp.state_constraints[k].F.transpose() * v;
    }
  }

  void step();
  void update_q_and_s();
  void update_Q_and_R();
  void riccati_factorization();
  void riccati_solve();
  bool converged();
};

} // namespace ocss

#include "details/projectors.hpp"
#include "details/riccati.hpp"
#include "details/solver.hpp"
