#include "MixtureModel.hpp"
#include <algorithm>

// constructor
MixtureModel::MixtureModel(arma::vec data, unsigned int k, unsigned int thin,
                           unsigned int burnin, unsigned int sample) {
    // assign data
    _data = data;    
    _n = _data.size();

    // assign MCMC parameters
    _nBurn = burnin;
    _nSample = sample;
    _nThin = thin;
    _k = k;
    _s = 0;
    _b = 0;

    // tuning parameters
    _delta_theta = 1.0;
    _delta_sigma = 0.5;

    // initial values to parameters from prior
    _theta = rnorm_cpp(_k, 0.0, 10.0);
    _sigma = arma::randg(_k, arma::distr_param(1, 1));
    _z = rz_cpp(_n, _k);

    // reallocate chains for parameters
    _theta_chain.resize(_nSample, _k);
    _sigma_chain.resize(_nSample, _k);
    _z_chain.resize(_nSample, _n);
}

// destructor (empty for now)
MixtureModel::~MixtureModel() {
}

// run burnin
void MixtureModel::run_burnin() {
    while (_b < _nBurn) {
        update_theta();
        update_sigma();
        update_z();
        _b++;
    }
}

// run burnin
void MixtureModel::posterior_sample() {
    while (_s < _nSample) {
        // update and save parameters
        update_theta(true);
        update_sigma(true);
        update_z(true);
        _s++;

        // run _nThin thinning intervals
        for (int i = 0; i < _nThin; ++i) {
            update_theta();
            update_sigma();
            update_z();
        }
    }
}

// get stored chains
std::map<std::string, arma::mat> MixtureModel::get_chains() {
    std::map<std::string, arma::mat> chains;
    chains["theta"] = _theta_chain;
    chains["sigma"] = _sigma_chain;
    chains["z"] = arma::conv_to<arma::mat>::from(_z_chain);

    return chains;
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
        theta_star = arma::conv_to<double>::from(rnorm_cpp(1, _theta[i], _delta_theta));

        // calculate log acceptance probability
        log_r = dnorm_cpp_vec(y_this_z, theta_star, 1. / _sigma[i]) + 
                dnorm_cpp(theta_star, 0, 10);
        log_r -= dnorm_cpp_vec(y_this_z, _theta[i], 1. / _sigma[i]) +
                 dnorm_cpp(_theta[i], 0, 10);

        // probabilistically accept theta_star
        if (log(arma::randu<double>()) < log_r) {
            _theta[i] = theta_star;
        }
    }

    _theta = arma::sort(_theta);

    // if burnin complete, save parameter in chain
    if (save) {
        _theta_chain.row(_s) = _theta.t();
    }
}

// update the variances of the components
void MixtureModel::update_sigma(bool save) {
    // initialize proposed value and log r
    double sigma_star;
    double log_r;

    for (int i = 0; i < _k; ++i) {
        // get only y's from this component
        arma::vec y_this_z = _data.elem(arma::find(_z == i));

        // propose new variances (precision actually)
        sigma_star = runif_cpp(std::max(0., _sigma[i] - _delta_sigma),
                               _sigma[i] + _delta_sigma);

        // calculate log acceptance probability
        log_r = dnorm_cpp_vec(y_this_z, _theta[i], 1. / sigma_star) + 
                dgamma_cpp(sigma_star, 1, 1) + 
                dunif_cpp(std::max(0., sigma_star - _delta_sigma),
                          sigma_star + _delta_sigma);
        log_r -= dnorm_cpp_vec(y_this_z, _theta[i], 1. / _sigma[i]) +
                 dgamma_cpp(_sigma[i], 1, 1) +
                 dunif_cpp(std::max(0., _sigma[i] - _delta_sigma),
                           _sigma[i] + _delta_sigma);

        // probabilistically accept theta_star
        if (log(arma::randu<double>()) < log_r) {
            _sigma[i] = sigma_star;
        }
    }

    // if burnin complete, save parameter in chain
    if (save) {
        _sigma_chain.row(_s) = _sigma.t();
    }
}

// update the latent assignments
void MixtureModel::update_z(bool save) {
    // initialize proposed value and log r
    arma::ivec z_star;
    double log_r;

    // propose new latent assignments
    z_star = rz_cpp(_n, _k);

    // iterate over all of the z's
    for (int i = 0; i < _n; ++i) {
        // calculate log acceptance probability
        log_r = dnorm_cpp(_data[i], _theta[z_star[i]], 1. / _sigma[z_star[i]]) -
                dnorm_cpp(_data[i], _theta[_z[i]], 1. / _sigma[_z[i]]);
        // NB: Q(z'|z) = Q(z|z') so this isn't factored into acceptance probability

        // probabilistically accept theta_star
        if ((log(arma::randu<double>()) < log_r) && (z_star[i] != _z[i])) {
            _z[i] = z_star[i];
        }
    }

    if (save) {
        _z_chain.row(_s) = _z.t();
    }
}
