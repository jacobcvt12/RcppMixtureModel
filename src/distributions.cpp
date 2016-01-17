#include "distributions.hpp"

arma::vec rnorm_cpp(int n, double mean, double variance) {
    arma::vec draws = arma::randn(n, 1);

    // multiply by sqrt of variance
    draws *= sqrt(variance);

    // add mean
    draws += mean;

    return draws;
}

double dnorm_cpp_vec(arma::vec y, double mean, double variance) {
    double n = y.size();
    double log_lik; 
    log_lik = -n * log(sqrt(variance)) - n * 0.5 * log(2 * M_PI) - 
              arma::sum(arma::square(y - mean)) / (2. * variance);
    
    return log_lik;
}

double dnorm_cpp(double y, double mean, double variance) {
    double log_lik; 
    log_lik = -log(sqrt(variance)) - 0.5 * log(2 * M_PI) - 
              pow(y - mean, 2.) / (2. * variance);
    
    return log_lik;
}

double dgamma_cpp(double y, double shape, double scale) {
    double log_lik = (shape - 1.) * log(y) - 1. / scale * y - 
                     shape * log(scale) - lgamma(shape);

    return log_lik;
}

double runif_cpp(double a, double b) {
    double u = arma::randu();

    return a + (b - a) * u;
}

double dunif_cpp(double a, double b) {
    return -log(b - a);
}

arma::vec rdirichlet_cpp(arma::vec alpha) {
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

arma::ivec rmultinom_cpp(int n, arma::vec p) {
    arma::vec p_sum = arma::cumsum(p);
    arma::vec draws = arma::randu(n);
    arma::ivec multi(p.size());
    multi.fill(0);

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

arma::ivec rz_cpp(unsigned int n, unsigned int k) {
    arma::ivec S(n);
    arma::vec prob_window(k);
    prob_window.fill(1. / (double) k);
    prob_window = arma::cumsum(prob_window);

    for (int i = 0; i < n; ++i) {
        S[i] = std::rand() % k;
    }

    return S;
}
