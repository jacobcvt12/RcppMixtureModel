#include "MixtureModel.hpp"

// [[Rcpp::export]]
Rcpp::List run_model(arma::vec data, unsigned int k, unsigned int thin,
                     unsigned int burnin, unsigned int sample) {
    MixtureModel model(data, k, thin, burnin, sample);

    // run MCMC
    model.run_burnin();
    model.posterior_sample();

    return model.get_chains();
}
