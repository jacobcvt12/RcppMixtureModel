#include "MixtureModel.hpp"

// [[Rcpp::export]]
Rcpp::List run_model(arma::vec data, unsigned int k, unsigned int thin,
                     unsigned int burnin, unsigned int sample) {
    MixtureModel model(data, k, thin, burnin, sample);

    // run MCMC
    model.run_burnin();
    model.posterior_sample();

    std::map<std::string, arma::mat> chains = model.get_chains();
    return Rcpp::List::create(Rcpp::Named("theta")=chains["theta"],
                              Rcpp::Named("sigma")=chains["sigma"],
                              Rcpp::Named("z")=chains["z"]);
}
