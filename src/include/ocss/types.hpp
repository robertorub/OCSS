#pragma once
#include <Eigen/Dense>
#include <array>

namespace ocss {

template<typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde>
struct OptimizationVariables{
  Eigen::Matrix<scalar, nx, N + 1> x; //std::vector<Vec> x;      // nx x (N+1)
  Eigen::Matrix<scalar, nu, N> u; //std::vector<Vec> u;      // nu x N
  Eigen::Matrix<scalar, nx_tilde, N + 1> x_tilde; //std::vector<Vec> x;     
  Eigen::Matrix<scalar, nu_tilde, N> u_tilde; //std::vector<Vec> u;     
  Eigen::Matrix<scalar, nx_tilde, N + 1> y_x; //std::vector<Vec> y_x; //dual  multiplier of xtilde
  Eigen::Matrix<scalar, nu_tilde, N> y_u; // std::vector<Vec> y_u; // dual multiplier of utilde
  Eigen::Matrix<scalar, nx_tilde, N + 1> x_tilde_prev; //std::vector<Vec> x;     
  Eigen::Matrix<scalar, nu_tilde, N> u_tilde_prev; //std::vector<Vec> u;
  

  void setZero() noexcept{
        x.setZero();
        u.setZero();
        x_tilde.setZero();
        u_tilde.setZero();
        y_x.setZero();
        y_u.setZero();
        x_tilde_prev.setZero();
        u_tilde_prev.setZero();
  }

  void reset() noexcept{
        setZero(); // usually it is only required to reset the dual variables y 
  }
};

template <typename scalar, int nx, int nu, int N>
struct RiccatiFactors {
  // Stage-indexed matrices:
  // Lambda[k] : nu x nu,   k = 0..N-1
  // P[k]      : nx x nx,   k = 0..N     (terminal at k=N)
  // L[k]      : nu x nx,   k = 0..N-1
  std::array<Eigen::Matrix<scalar, nu, nu>, N> Lambda;
  std::array<Eigen::Matrix<scalar, nx, nx>, N + 1> P;
  std::array<Eigen::Matrix<scalar, nu, nx>, N> L;

  // Stage-indexed vectors stored column-wise:
  // l : nu x N     (col k is l_k)
  // p : nx x (N+1) (col k is p_k, terminal at col N)
  Eigen::Matrix<scalar, nu, N> l; //std::vector<Vec> l;
  Eigen::Matrix<scalar, nx, N + 1> p; //std::vector<Vec> p;

  // ---------- Temporary buffers for riccati factorization (preallocated) ----------
  Eigen::Matrix<scalar, nx, nx> pa;    // nx x nx
  Eigen::Matrix<scalar, nx, nu> pb;    // nx x nu
  //Eigen::Matrix<scalar, nu, nx> bpa;   // nu x nx
  Eigen::Matrix<scalar, nu, nu> bpb;   // nu x nu
  Eigen::Matrix<scalar, nx, nx> apa;   // nx x nx
  void setZero() noexcept {
    for(int i=0; i<N; ++i){
        Lambda[i].setZero();
        L[i].setZero();
        P[i].setZero();
    };
    P[N].setZero();
    l.setZero();
    p.setZero();
    pa.setZero();
    pb.setZero();
    //bpa.setZero();
    bpb.setZero();
    apa.setZero();
  }


};

template <typename scalar, int nx, int nu, int N, int nx_tilde, int nu_tilde>
struct ADMMWorkspace {
    RiccatiFactors<scalar, nx, nu, N> factors;
    OptimizationVariables<scalar, nx, nu, N, nx_tilde, nu_tilde> variables;
    Eigen::Matrix<scalar, nx, 1> tmp_x;
    Eigen::Matrix<scalar, nx_tilde, 1> tmp1_x_tilde;
    Eigen::Matrix<scalar, nx_tilde, 1> tmp2_x_tilde;
    Eigen::Matrix<scalar, nu, 1> tmp_u;
    Eigen::Matrix<scalar, nu_tilde, 1> tmp1_u_tilde;
    Eigen::Matrix<scalar, nu_tilde, 1> tmp2_u_tilde;

    std::array<Eigen::Matrix<scalar, nx, nx>, N> Q_admm;
    std::array<Eigen::Matrix<scalar, nu, nu>, N> R_admm;
    Eigen::Matrix<scalar, nx, nx> Pn_admm;
    Eigen::Matrix<scalar, nx, N> q_admm;
    Eigen::Matrix<scalar, nu, N> s_admm;
    Eigen::Matrix<scalar, nx, 1> pn_admm;

    void setZero() noexcept{
        factors.setZero();
        variables.setZero();
        tmp_x.setZero();
        tmp1_x_tilde.setZero();
        tmp2_x_tilde.setZero();
        tmp_u.setZero();
        tmp1_u_tilde.setZero();
        tmp2_u_tilde.setZero();
        for(int i=0; i<N; ++i){
            Q_admm[i].setZero();
            R_admm[i].setZero();
        }
        Pn_admm.setZero();
        q_admm.setZero();
        s_admm.setZero();
        pn_admm.setZero();
    }
};

template <typename scalar>
struct Status {
    int iterations = 0;
    scalar primal_residual = scalar(0);
    scalar dual_residual = scalar(0);
    bool converged = false;
};

}