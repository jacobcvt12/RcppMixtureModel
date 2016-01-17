#include "MixtureModel.hpp"

// [[Rcpp::export]]
Rcpp::List run_model(arma::vec data, unsigned int k, unsigned int thin,
                     unsigned int burnin, unsigned int sample) {
    MixtureModel model(data, k, thin, burnin, sample);

    // run MCMC
    model.run_burnin();
    model.posterior_sample();

    // get chains and return them
    std::map<std::string, arma::mat> chains = model.get_chains();
    Rcpp::List chains_list;
    chains_list["theta"] = chains["theta"];
    chains_list["sigma"] = chains["sigma"];
    chains_list["z"] = chains["z"];
        
    return chains_list;
}
