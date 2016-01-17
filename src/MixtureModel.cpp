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
    _s = 0;
    _b = 0;

    // tuning parameters
    _delta_theta = 2.5;

    // initial values to parameters from prior
    _theta = rnorm(_k, 0.0, 1000.0);
    _sigma = arma::vec(std::vector<double> {2.5, 2.5, 2.5}); // consider known
    _lambda = arma::vec(std::vector<double> {0.2, 0.3, 0.5}); // consider known
    _z = rz(_n, _k);

    // reallocate chains for parameters
    _theta_chain.resize(_nSample, _k);
    _sigma_chain.resize(_nSample, _k);
    _lambda_chain.resize(_nSample, _k);
}

// update the means of the components
void MixtureModel::update_theta(bool save) {
    // initialize proposed value and log r
    double theta_star;
    double log_r;

    for (int i = 0; i < _k; ++i) {
        // get only y's from this component
        arma::vec y_this_z = _data.elem(arma::find(_z == i));

        // propose new thetas
        theta_star = arma::conv_to<double>::from(rnorm(1, _theta[i], _delta_theta));

        // calculate log acceptance probability
        log_r = dnorm(y_this_z, theta_star, _sigma[i]) + 
                dnorm(theta_star, 0, 100);
        log_r -= dnorm(y_this_z, _theta[i], _sigma[i]) +
                 dnorm(_theta[i], 0, 100);

        // probabilistically accept theta_star
        if (log(arma::randu<double>()) < log_r) {
            _theta[i] = theta_star;
        }

        // if burnin complete, save parameter in chain
        if (save) {
            _theta_chain[_s, i] = _theta[i];
        }
    }
}

// update the latent assignments
void MixtureModel::update_z() {
    // initialize proposed value and log r
    arma::ivec z_star;
    double log_r;

    for (int i = 0; i < _k; ++i) {
        // propose new latent assignments
        z_star = rmultinom(_n, _lambda);

        //// calculate log acceptance probability
        //log_r = dnorm(y_this_z, theta_star, _sigma[i]) + 
                //dnorm(theta_star, _theta[i], _delta_theta);
        //log_r -= dnorm(y_this_z, _theta[i], _sigma[i]) +
                 //dnorm(_theta[i], theta_star, _delta_theta);

        //// probabilistically accept theta_star
        //if (log(arma::randu<double>()) < log_r) {
            //_theta[i] = theta_star;
        //}
    }
}
