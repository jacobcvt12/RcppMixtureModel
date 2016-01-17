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
    _delta_theta = 1.0;

    // initial values to parameters from prior
    _theta = rnorm_cpp(_k, 0.0, 10.0);
    _sigma = arma::randg(_k, arma::distr_param(1, 1));
    _lambda = rdirichlet_cpp(arma::vec(_k).fill(1));
    _z = rz_cpp(_n, _k);

    // reallocate chains for parameters
    _theta_chain.resize(_nSample, _k);
    _sigma_chain.resize(_nSample, _k);
    _lambda_chain.resize(_nSample, _k);
}

MixtureModel::MixtureModel(arma::vec data, unsigned int k,
                           arma::vec sigma, arma::vec lambda, arma::ivec z,
                           unsigned int burnin, unsigned int sample) :
    MixtureModel(data, k, burnin, sample) {
    _sigma = sigma;
    _lambda = lambda;
    _z = z;
}

// destructor (empty for now)
MixtureModel::~MixtureModel() {
}

// run burnin
void MixtureModel::run_burnin() {
    while (_b < _nBurn) {
        update_theta();
        _b++;
    }
}

// run burnin
void MixtureModel::posterior_sample() {
    while (_s < _nSample) {
        update_theta(true);
        _s++;
    }
}

// get stored chains
Rcpp::List MixtureModel::get_chains() {
    return Rcpp::List::create(Rcpp::Named("theta")=_theta_chain);
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
        log_r = dnorm_cpp_vec(y_this_z, theta_star, _sigma[i]) + 
                dnorm_cpp(theta_star, 0, 10);
        log_r -= dnorm_cpp_vec(y_this_z, _theta[i], _sigma[i]) +
                 dnorm_cpp(_theta[i], 0, 10);

        // probabilistically accept theta_star
        if (log(arma::randu<double>()) < log_r) {
            _theta[i] = theta_star;
        }

        // if burnin complete, save parameter in chain
        if (save) {
            _theta_chain(_s, i) = _theta[i];
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
        z_star = rmultinom_cpp(_n, _lambda);

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
