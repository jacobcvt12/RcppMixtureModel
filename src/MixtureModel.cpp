#include "MixtureModel.hpp"

// constructor
MixtureModel::MixtureModel(arma::vec data, unsigned int k,
                           unsigned int burnin, unsigned int sample) {
    // assign data
    _data = data;    
    _n = _data.size();

    // assign MCMC parameters
    _nBurn = burnin;
    _nSample = sample;
    _k = k;

    // initial values to parameters from prior
    _theta = rnorm(_k, 0.0, 1000.0);
    _sigma = arma::vec(std::vector<double> {2.5, 2.5, 2.5}); // consider known
    _lambda = arma::vec(std::vector<double> {0.2, 0.3, 0.5}); // consider known
    _z = rz(_n, _k);
}

