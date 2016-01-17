#include "MixtureModel.hpp"

// [[Rcpp::export]]
Rcpp::List run_model(arma::vec data, unsigned int k,
                     arma::vec sigma, arma::vec lambda, arma::ivec z,
                     unsigned int burnin, unsigned int sample) {
    MixtureModel model(data, k, sigma, lambda, z, burnin, sample);

    // run MCMC
    model.run_burnin();
    model.posterior_sample();

    return model.get_chains();
}
