#pragma once
#include "details/projectors.hpp"
#include <Eigen/Dense>
#include <array>

namespace ocss {
template <typename T, int N, bool UniformTrajectoryConstraints>
struct TrajectoryConstraints;

template <typename T, int N> struct TrajectoryConstraints<T, N, true> {
  T constraints;
  TrajectoryConstraints() = default;
  TrajectoryConstraints(const T &c) : constraints(c) {}
  T &operator[](int) { return constraints; }
  const T &operator[](int) const { return constraints; }
};

template <typename T, int N> struct TrajectoryConstraints<T, N, false> {
  std::array<T, N> constraints;
  T &operator[](int i) { return constraints[i]; }
  const T &operator[](int i) const { return constraints[i]; }
};

// Forward declaration
template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
class ADMM;

template <typename scalar, int nx, int nu, int _N, int nx_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
class OptimalControlData {
  static constexpr int N = _N;
  // [x u][ Q 0; 0 R] [x; u] + [q; s]'[x; u] + xPNx + pn'x
  // x+ = Ax + Bu + b

  // Friend class declaration
  template <typename, int, int, int, int, int, bool, bool, typename,
            typename...>
  friend class ADMM;

  // Dynamics & costs (stage-wise)
private:
  std::array<Eigen::Matrix<scalar, nx, nx>, N> Ad;
  std::array<Eigen::Matrix<scalar, nx, nu>, N> Bd;
  std::array<Eigen::Matrix<scalar, nx, nx>, N> Q;
  std::array<Eigen::Matrix<scalar, nu, nu>, N> R;
  // std::array<Eigen::Matrix<scalar, nx, nu>, N> S; // cross-term
  //  Terminal cost
  Eigen::Matrix<scalar, nx, nx> Pn;
  // Optional vectors/matrices
  Eigen::Matrix<scalar, nx, N> b, q;
  Eigen::Matrix<scalar, nu, N> s;

  Eigen::Matrix<scalar, nx, 1> pn;

  // Input constraints
  TrajectoryConstraints<InputSet, N, UniformTrajectoryInputConstraints>
      input_constraints;
  // InputSet input_constraints_;
  //  State constraints
  TrajectoryConstraints<StageConstraints<scalar, nx, nx_tilde, Projectors...>,
                        N, UniformTrajectoryStateConstraints>
      state_constraints;
  // StageConstraints<scalar, nx, nx_tilde, Projectors...> state_constraints_;
public:
  void set_Ad(const std::array<Eigen::Matrix<scalar, nx, nx>, N> &_Ad) {
    Ad = _Ad;
  }

  void set_Bd(const std::array<Eigen::Matrix<scalar, nx, nu>, N> &_Bd) {
    Bd = _Bd;
  }

  void set_Q(const std::array<Eigen::Matrix<scalar, nx, nx>, N> &_Q) { Q = _Q; }

  void set_R(const std::array<Eigen::Matrix<scalar, nu, nu>, N> &_R) { R = _R; }

  void set_Pn(const Eigen::Matrix<scalar, nx, nx> &_Pn) { Pn = _Pn; }

  void set_b(const Eigen::Matrix<scalar, nx, N> &_b) { b = _b; }

  void set_q(const Eigen::Matrix<scalar, nx, N> &_q) { q = _q; }

  void set_s(const Eigen::Matrix<scalar, nu, N> &_s) { s = _s; }

  void set_pn(const Eigen::Matrix<scalar, nx, 1> &_pn) { pn = _pn; }

  // updating the whole constraints set is not efficient, TODO allow to modify
  // only internal parameters
  void set_input_constraints(
      const TrajectoryConstraints<
          InputSet, N, UniformTrajectoryInputConstraints> &_input_constraints) {
    input_constraints = _input_constraints;
  }

  void set_state_constraints(
      const TrajectoryConstraints<
          StageConstraints<scalar, nx, nx_tilde, Projectors...>, N,
          UniformTrajectoryStateConstraints> &_state_constraints) {
    state_constraints = _state_constraints;
  }

  // allow the user to modify only what is needed in the input constraints
  auto &modify_input_constraint() { return input_constraints.constraints; }

  auto &modify_state_constraint() { return state_constraints.constraints; }

  auto &modify_Ad() { return Ad; }

  auto &modify_Bd() { return Bd; }

  auto &modify_Q() { return Q; }

  auto &modify_R() { return R; }

  auto &modify_Pn() { return Pn; }

  auto &modify_q() { return q; }

  auto &modify_pn() { return pn; }
};

} // namespace ocss