#pragma once
#include "ocss/types.hpp"
#include <Eigen/Dense>

namespace ocss {

template <typename scalar, int _dim, int offset> struct BoxData {
  static constexpr int dim = _dim;
  Eigen::Matrix<scalar, dim, 1> lb, ub;
  template <typename DerivedY, typename DerivedX>
  void project(Eigen::MatrixBase<DerivedY> &y,
               const Eigen::MatrixBase<DerivedX> &x) const {

    y.derived().template segment<dim>(offset) =
        x.derived().template segment<dim>(offset).cwiseMax(lb).cwiseMin(ub);
  }
};

template <typename scalar, int _dim, int offset> struct BallData {
  static constexpr int dim = _dim;
  scalar radius;
  template <typename DerivedY, typename DerivedX>
  void project(Eigen::MatrixBase<DerivedY> &y,
               const Eigen::MatrixBase<DerivedX> &x) const {
    scalar norm = x.derived().template segment<dim>(offset).norm();
    if (norm > radius) {
      y.derived().template segment<dim>(offset) =
          (radius / norm) * x.derived().template segment<dim>(offset);
    } else {
      y.derived().template segment<dim>(offset) =
          x.derived().template segment<dim>(offset);
    }
  }
};

template <typename scalar, int _dim, int offset> struct ComplementarityData {
  // TO DO
  // x >= 0
  // y >= 0
  // x^T y = 0
};

template <typename scalar, int _dim, int offset> struct SphericalObstacleData {
  // TODO
  // norm_2(x -c) >= d
};

template <typename scalar, int _dim, int offset> struct PolytopeObstacleData {
  // TODO
  // for all p \in convex_hull_points h' p >= d_min
  // for all s in trajectory h' s <= d_min
  // projection alternate between h and s
};

template <typename scalar, int _dim, int offset> struct NormData {
  // TODO
  // norm_2(x) = d
};

template <typename scalar, int nx, int nx_tilde, typename... Projectors>
struct StageConstraints {
  static_assert((Projectors::dim + ...) == nx_tilde,
                "Sum of projector dimensions must equal nx_tilde");
  Eigen::Matrix<scalar, nx, nx> FTF;
  Eigen::Matrix<scalar, nx_tilde, nx> F;
  std::tuple<Projectors...> projectors;
  template <typename DerivedY, typename DerivedX>
  void apply_projections(Eigen::MatrixBase<DerivedY> &y,
                         const Eigen::MatrixBase<DerivedX> &x) const {
    std::apply([&y, &x](auto &&...proj) { (..., proj.project(y, x)); },
               projectors);
  }
  void apply_F(Eigen::Matrix<scalar, nx_tilde, 1> &y,
               const Eigen::Matrix<scalar, nx, 1> &x) const {
    // todo implement efficient kernel for F
  }

  void apply_FT(Eigen::Ref<Eigen::Matrix<scalar, nx, 1>> y,
                const Eigen::Ref<Eigen::Matrix<scalar, nx_tilde, 1>> x) const {
    // todo implement efficient kernel for FT
  }

  void apply_FTF(Eigen::Matrix<scalar, nx, 1> &y,
                 const Eigen::Matrix<scalar, nx, 1> &x) const {
    // todo implement efficient kernel for FTF
  }
};

} // namespace ocss