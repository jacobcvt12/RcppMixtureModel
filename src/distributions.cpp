#include "distributions.hpp"

arma::vec rnorm(int n, double mean, double variance) {
    arma::vec draws = arma::randn(n, 1);

    // multiply by sqrt of variance
    draws *= sqrt(variance);

    // add mean
    draws += mean;

    return draws;
}

arma::vec rdirichlet(int n, arma::vec alpha) {
    arma::vec Y(alpha.size());
    arma::vec X(alpha.size());

    // draw Y_k ~ gamma(a_k, 1)
    for (int k = 0; k < alpha.size(); ++k) {
        double a = alpha[k];
        double theta = 1.;
        Y[k] = arma::conv_to<double>::from(arma::randg(1, arma::distr_param(a, theta)));
    }

    // calculate V ~ gamma(\sum alpha_i, 1)
    double V = arma::sum(Y);

    // calculate X ~ Dir(alpha_1, ..., alpha_k)
    for (int k = 0; k < alpha.size(); ++k) {
        X[k] = Y[k] / V;
    }

    return X;
}

std::vector<int> rmultinom(int n, arma::vec p) {
    arma::vec p_sum = arma::cumsum(p);
    arma::vec draws = arma::randu(n);
    std::vector<int> multi(p.size());

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < p.size(); ++k) {
            if (draws[i] <= p_sum[k]) {
                multi[k]++;
                break;
            }
        }
    }

    return multi;
}

std::vector<int> sample(int n, int k) {
    std::vector<int> S(n);
    arma::vec prob_window(k);
    prob_window.fill(1. / (double) k);
    prob_window = arma::cumsum(prob_window);

    for (int i = 0; i < n; ++i) {
        S[i] = std::rand() % k;
    }

    return S;
}
