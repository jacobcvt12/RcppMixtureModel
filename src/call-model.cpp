#include "MixtureModel.hpp"
#include <omp.h>

// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export]]
Rcpp::List run_model(arma::vec data, unsigned int k, unsigned int thin,
                     unsigned int burnin, unsigned int sample,
                     unsigned int cores) {
    arma::cube theta(sample, k, cores);
    arma::cube sigma(sample, k, cores);
    arma::cube z(sample, data.size(), cores);

    #pragma omp parallel for num_threads(cores)
    for (int i = 0; i < cores; ++i) {
        // initialize model
        MixtureModel model(data, k, thin, burnin, sample);

        // run MCMC
        model.run_burnin();
        model.posterior_sample();
        
        // get chains and return them
        std::map<std::string, arma::mat> chains = model.get_chains();
        theta.slice(i) = chains["theta"];
        sigma.slice(i) = chains["sigma"];
        z.slice(i) = chains["z"];
    }

    // create list to store chains of parameter estimates
    Rcpp::List chains_list;
    chains_list["theta"] = theta;
    chains_list["sigma"] = sigma;
    chains_list["z"] = z;

    return chains_list;
}
