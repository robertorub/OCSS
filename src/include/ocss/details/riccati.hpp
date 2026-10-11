#pragma once
#include "ocss/types.hpp"
#include <Eigen/Dense>

namespace ocss {

template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
void ADMM<scalar, nx, nu, N, nx_tilde, nu_tilde,
          UniformTrajectoryStateConstraints, UniformTrajectoryInputConstraints,
          InputSet, Projectors...>::riccati_factorization() {
  // from Efficient Implementation of the Riccati Recursion for Solving
  // Linear-Quadratic Control Problems Frison 2013

  // Terminal condition: P[N] = Pn
  ws.factors.P[N] = ws.Pn_admm;

  // Backward Riccati recursion
  auto &fac = ws.factors;
  for (int k = N - 1; k >= 0; --k) {

    // G = [A B]' P_{k+1} [A B] = [A'PA A'PB; B'PA B'PB]
    fac.AB.template leftCols<nx>() = ocp.Ad[k];
    fac.AB.template rightCols<nu>() = ocp.Bd[k];
    fac.PAB.noalias() = fac.P[k + 1] * fac.AB;
    fac.G.noalias() = fac.AB.transpose() * fac.PAB;

    // P_{k+1} b_k is constant during the ADMM iterations
    fac.Pb.col(k).noalias() = fac.P[k + 1] * ocp.b.col(k);

    // Lambda Lambda' = B'PB + R (symmetrized)
    fac.Lambda[k] = fac.G.template bottomRightCorner<nu, nu>() + ws.R_admm[k];
    fac.Lambda[k] = (scalar(0.5) * (fac.Lambda[k] + fac.Lambda[k].transpose())).eval();

    // in place cholesky, only the lower part is updated (the upper part is unusable)
    Eigen::LLT<Eigen::Ref<Eigen::Matrix<scalar, nu, nu>>> llt(fac.Lambda[k]);
    fac.Linv[k] = fac.Lambda[k].template triangularView<Eigen::Lower>().solve(
        Eigen::Matrix<scalar, nu, nu>::Identity());

    // L_k = Lambda^{-1} B'PA
    fac.L[k].noalias() = fac.Linv[k] * fac.G.template bottomLeftCorner<nu, nx>();

    // P_k = A'PA + Q - L'L (symmetrized)
    fac.P[k] = fac.G.template topLeftCorner<nx, nx>() + ws.Q_admm[k];
    fac.P[k].noalias() -= fac.L[k].transpose() * fac.L[k];
    fac.P[k] = (scalar(0.5) * (fac.P[k] + fac.P[k].transpose())).eval();
  }
}

template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde,
          bool UniformTrajectoryStateConstraints,
          bool UniformTrajectoryInputConstraints, typename InputSet,
          typename... Projectors>
void ADMM<scalar, nx, nu, N, nx_tilde, nu_tilde,
          UniformTrajectoryStateConstraints, UniformTrajectoryInputConstraints,
          InputSet, Projectors...>::riccati_solve() {

  // p[:, N] = ocp.pn   (terminal affine term)
  ws.factors.p.col(N) = ws.pn_admm;

  // -------- Backward pass: compute l[k], p[k] for k=N-1..0 --------
  for (int k = N - 1; k >= 0; --k) {
    auto pnext = ws.factors.p.col(k + 1);
    auto pn = ws.factors.p.col(k);
    auto ln = ws.factors.l.col(k);
    const Eigen::Matrix<scalar, nx, nx> &An = ocp.Ad[k];
    const Eigen::Matrix<scalar, nx, nu> &Bn = ocp.Bd[k];
    auto qn = ws.q_admm.col(k);
    auto sn = ws.s_admm.col(k);
    const Eigen::Matrix<scalar, nu, nx> &Ln = ws.factors.L[k]; // nu×nx

    // tmp_x = Pnext' * b + pnext
    ws.tmp_x = ws.factors.Pb.col(k);
    ws.tmp_x.noalias() += pnext;

    // l[k] = Lam^{-1} (B' * tmp_x + s)
    ws.tmp_u.noalias() = Bn.transpose() * ws.tmp_x;
    ws.tmp_u.noalias() += sn;
    ln.noalias() = ws.factors.Linv[k] * ws.tmp_u;

    // p[k] = A' * tmp_x + q - Lk' * l[k]
    pn.noalias() = An.transpose() * ws.tmp_x;
    pn.noalias() += qn;
    pn.noalias() -= Ln.transpose() * ln;
  }

  // -------- Initialize lambda[0] = P0*x0 + p0 --------
  // sol.lambda[0].noalias() = factors.P[0] * sol.x[0]; // unused
  // sol.lambda[0].noalias() += factors.p[0]; // unused

  // -------- Forward pass: compute u[k], x[k+1], lambda[k+1] --------
  for (int k = 0; k < N; ++k) {
    const Eigen::Matrix<scalar, nx, nx> &Ak = ocp.Ad[k];
    const Eigen::Matrix<scalar, nx, nu> &Bk = ocp.Bd[k];
    const auto bk = ocp.b.col(k);
    const Eigen::Matrix<scalar, nx, nx> &Pnext = ws.factors.P[k + 1];
    const auto pnext = ws.factors.p.col(k + 1);
    const Eigen::Matrix<scalar, nu, nx> &Lk = ws.factors.L[k]; // nu×nx
    const auto lk = ws.factors.l.col(k);

    auto xk = ws.variables.x.col(k);
    auto xnext = ws.variables.x.col(k + 1);
    auto uk = ws.variables.u.col(k);
    // Vec& lambda_next  = sol.lambda[k + 1]; // unused

    // u = -Lam^{-T} (L x + l[k])
    ws.tmp_u.noalias() = Lk * xk;
    ws.tmp_u.noalias() += lk;
    uk.noalias() = -(ws.factors.Linv[k].transpose() * ws.tmp_u);

    // xnext = A*xk + b + B*u
    xnext.noalias() = Ak * xk;
    xnext.noalias() += bk;
    xnext.noalias() += Bk * uk;

    // lambda_next.noalias() = Pnext * xnext; // unused
    // lambda_next.noalias() += pnext; // unused
  }
}

} // namespace ocss