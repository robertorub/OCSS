#pragma once

namespace ocss {
template <typename scalar> struct Options {
public:
  scalar rho = scalar(0.5);
  scalar alpha = scalar(1.6);

  scalar abs_tol = scalar(1e-3);
  scalar rel_tol = scalar(1e-3);
  int max_iterations = 1000;
  scalar rho_min = scalar(0.1);
};
} // namespace ocss