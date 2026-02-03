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
  for (int k = N - 1; k >= 0; --k) {

    // pa = P_{k+1}' * A_k
    ws.factors.pa.noalias() = ws.factors.P[k + 1] * ocp.Ad[k];

    // pb = P_{k+1}' * B_k
    ws.factors.pb.noalias() = ws.factors.P[k + 1] * ocp.Bd[k];

    // bpa = B_k' * pa
    // ws.factors.bpa.noalias() = ocp.Bd[k].transpose() * ws.factors.pa;
    ws.factors.L[k].noalias() = ocp.Bd[k].transpose() * ws.factors.pa;
    // bpb = B_k' * pb + R_k
    ws.factors.bpb.noalias() = ocp.Bd[k].transpose() * ws.factors.pb;
    ws.factors.bpb.noalias() += ws.R_admm[k];
    // Symmetrize
    // ws.factors.bpb = 0.5 * (ws.factors.bpb + ws.factors.bpb.transpose());
    ws.factors.Lambda[k] = 0.5 * (ws.factors.bpb + ws.factors.bpb.transpose());

    // Cholesky: bpb = L * L'

    // Eigen::LLT<Eigen::Matrix<scalar, nu, nu>> llt(ws.factors.bpb);

    // std::cout << "Riccati factorization k= " << k << "\n";

    Eigen::LLT<Eigen::Ref<Eigen::Matrix<scalar, nu, nu>>> llt(
        ws.factors.Lambda[k]); // in place cholesky, note only the lower part is
                               // updated the upper part remains unchanged (and
                               // unusable!!!)

    // L_k = Λ_k^{-1} * bpa
    // ws.factors.L[k] =
    // ws.factors.Lambda[k].triangularView<Eigen::Lower>().solve(ws.factors.bpa);
    ws.factors.Lambda[k].template triangularView<Eigen::Lower>().solveInPlace(
        ws.factors.L[k]);

    // apa = A_k' * pa + Q_k
    ws.factors.apa.noalias() = ocp.Ad[k].transpose() * ws.factors.pa;
    ws.factors.apa.noalias() += ws.Q_admm[k];

    // apa -= L_k' * L_k
    ws.factors.apa.noalias() -=
        ws.factors.L[k].transpose() *
        ws.factors
            .L[k]; // better way, do a rank update as the matrix is symmetric

    // P_k = Symmetric(apa)
    ws.factors.P[k] = 0.5 * (ws.factors.apa + ws.factors.apa.transpose());
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
    Eigen::Matrix<scalar, nx, nx> &Pnext = ws.factors.P[k + 1];
    auto pnext = ws.factors.p.col(k + 1);
    auto pn = ws.factors.p.col(k);
    auto ln = ws.factors.l.col(k);
    const Eigen::Matrix<scalar, nx, nx> &An = ocp.Ad[k];
    const Eigen::Matrix<scalar, nx, nu> &Bn = ocp.Bd[k];
    const Eigen::Matrix<scalar, nx, 1> &bn = ocp.b.col(k);
    auto qn = ws.q_admm.col(k);
    auto sn = ws.s_admm.col(k);
    const Eigen::Matrix<scalar, nu, nu> &Lambdan =
        ws.factors.Lambda[k];                                  // lower
    const Eigen::Matrix<scalar, nu, nx> &Ln = ws.factors.L[k]; // nu×nx

    // tmp_x = Pnext' * b + pnext
    ws.tmp_x.noalias() = Pnext * bn;
    ws.tmp_x.noalias() += pnext;

    // l[k] = B' * tmp_x + s
    ln.noalias() = Bn.transpose() * ws.tmp_x;
    ln.noalias() += sn;

    // Solve Lam * l = l  (in-place lower-triangular solve)
    Lambdan.template triangularView<Eigen::Lower>().solveInPlace(ln);

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
    const Eigen::Matrix<scalar, nu, nu> &Lambdak =
        ws.factors.Lambda[k];                                  // lower
    const Eigen::Matrix<scalar, nu, nx> &Lk = ws.factors.L[k]; // nu×nx
    const auto lk = ws.factors.l.col(k);

    auto xk = ws.variables.x.col(k);
    auto xnext = ws.variables.x.col(k + 1);
    auto uk = ws.variables.u.col(k);
    // Vec& lambda_next  = sol.lambda[k + 1]; // unused

    ws.tmp_u.noalias() = Lk * xk;

    // u = -tmp_u - l[k]
    uk.noalias() = -ws.tmp_u;
    uk.noalias() -= lk;

    // Solve Lam^T * u = u  (in-place)
    Lambdak.transpose().template triangularView<Eigen::Upper>().solveInPlace(
        uk);

    // xnext = A*xk + b + B*u
    xnext.noalias() = Ak * xk;
    xnext.noalias() += bk;
    xnext.noalias() += Bk * uk;

    // lambda_next.noalias() = Pnext * xnext; // unused
    // lambda_next.noalias() += pnext; // unused
  }
}

} // namespace ocss