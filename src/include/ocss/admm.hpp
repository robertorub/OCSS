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

private:
  const OptimalControlData<
      scalar, nx, nu, N, nx_tilde, UniformTrajectoryStateConstraints,
      UniformTrajectoryInputConstraints, InputSet, Projectors...> &ocp;
  Options<scalar> options;
  ADMMWorkspace<scalar, nx, nu, N, nx_tilde, nu_tilde> ws;
  Status<scalar> status;
  scalar rho_scaling = scalar(1.0);
  scalar rho = options.rho;
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
