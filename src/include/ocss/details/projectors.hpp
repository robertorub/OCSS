#pragma once
#include "ocss/types.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>

namespace ocss {

// Soft sets penalize their violation d with w1 d + 0.5 w2 d^2; the ADMM
// z-update then uses the proximal operator prox(y, x, rho) of that penalty
// instead of the projection. w1 = +inf makes the set hard.
namespace detail {
template <typename P, typename = void> struct is_soft : std::false_type {};
template <typename P>
struct is_soft<P, std::void_t<decltype(P::is_soft)>>
    : std::bool_constant<P::is_soft> {};
} // namespace detail

template <typename P, typename DY, typename DX, typename scalar>
inline void prox_or_project(const P &p, Eigen::MatrixBase<DY> &y,
                            const Eigen::MatrixBase<DX> &x, scalar rho) {
  if constexpr (detail::is_soft<P>::value)
    p.prox(y, x, rho);
  else
    p.project(y, x);
}

template <typename P, typename DZ>
inline typename DZ::Scalar violation_of(const P &p,
                                        const Eigen::MatrixBase<DZ> &z) {
  if constexpr (detail::is_soft<P>::value)
    return p.violation(z);
  else
    return typename DZ::Scalar(0);
}

template <typename scalar, int _dim, int offset> struct BoxData {
  static constexpr int dim = _dim;
  static constexpr int start = offset;
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
  static constexpr int start = offset;
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

// Box with per-row soft bounds, d_i = max(0, z_i - ub_i, lb_i - z_i).
template <typename scalar, int _dim, int offset> struct SoftBoxData {
  static constexpr int dim = _dim;
  static constexpr int start = offset;
  static constexpr bool is_soft = true;
  using Vec = Eigen::Matrix<scalar, dim, 1>;
  Vec lb, ub;
  Vec w1 = Vec::Constant(std::numeric_limits<scalar>::infinity());
  Vec w2 = Vec::Zero();
  template <typename DerivedY, typename DerivedX>
  void project(Eigen::MatrixBase<DerivedY> &y,
               const Eigen::MatrixBase<DerivedX> &x) const {
    y.derived().template segment<dim>(offset) =
        x.derived().template segment<dim>(offset).cwiseMax(lb).cwiseMin(ub);
  }
  template <typename DerivedY, typename DerivedX>
  void prox(Eigen::MatrixBase<DerivedY> &y,
            const Eigen::MatrixBase<DerivedX> &x, scalar rho) const {
    for (int i = 0; i < dim; ++i) {
      const scalar v = x.derived()(offset + i);
      scalar z = v;
      if (v > ub(i)) {
        z = ub(i);
        if (std::isfinite(w1(i)))
          z = std::max(ub(i), (rho * v + w2(i) * ub(i) - w1(i)) / (rho + w2(i)));
      } else if (v < lb(i)) {
        z = lb(i);
        if (std::isfinite(w1(i)))
          z = std::min(lb(i), (rho * v + w2(i) * lb(i) + w1(i)) / (rho + w2(i)));
      }
      y.derived()(offset + i) = z;
    }
  }
  template <typename DerivedZ>
  scalar violation(const Eigen::MatrixBase<DerivedZ> &z) const {
    const auto zs = z.derived().template segment<dim>(offset);
    return std::max(scalar(0),
                    std::max((zs - ub).maxCoeff(), (lb - zs).maxCoeff()));
  }
};

// Soft version of a convex set P, d = Euclidean distance to the set.
// With p = P(x): z = p + (d' / d)(x - p), d' = max(0, (rho d - w1) / (rho + w2)).
template <typename scalar, typename P> struct Soft : P {
  static constexpr int dim = P::dim;
  static constexpr int start = P::start;
  static constexpr bool is_soft = true;
  scalar w1 = std::numeric_limits<scalar>::infinity();
  scalar w2 = scalar(0);
  template <typename DerivedY, typename DerivedX>
  void prox(Eigen::MatrixBase<DerivedY> &y,
            const Eigen::MatrixBase<DerivedX> &x, scalar rho) const {
    P::project(y, x);
    if (!std::isfinite(w1))
      return;
    auto ys = y.derived().template segment<dim>(start);
    const auto xs = x.derived().template segment<dim>(start);
    const scalar d = (xs - ys).norm();
    if (!(d > scalar(0)))
      return;
    const scalar dn = std::max(scalar(0), (rho * d - w1) / (rho + w2));
    ys += (dn / d) * (xs - ys);
  }
  template <typename DerivedZ>
  scalar violation(const Eigen::MatrixBase<DerivedZ> &z) const {
    typename DerivedZ::PlainObject p = z.derived();
    P::project(p, z);
    return (z.derived().template segment<dim>(start) -
            p.template segment<dim>(start))
        .norm();
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
  template <typename DerivedY, typename DerivedX>
  void apply_prox(Eigen::MatrixBase<DerivedY> &y,
                  const Eigen::MatrixBase<DerivedX> &x, scalar rho) const {
    std::apply(
        [&](auto &&...proj) { (..., prox_or_project(proj, y, x, rho)); },
        projectors);
  }
  // largest violation of the hard sets, over the soft projectors
  template <typename DerivedZ>
  scalar violation(const Eigen::MatrixBase<DerivedZ> &z) const {
    scalar v = scalar(0);
    std::apply(
        [&](auto &&...proj) { (..., (v = std::max(v, violation_of(proj, z)))); },
        projectors);
    return v;
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