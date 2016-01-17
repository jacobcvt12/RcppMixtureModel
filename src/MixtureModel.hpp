#ifndef _MIXTUREMODEL_H_
#define _MIXTUREMODEL_H_

#include "distributions.hpp"
#include <RcppArmadillo.h>
#include <vector>
#include <iostream>
#include <algorithm>
// [[Rcpp::depends(RcppArmadillo)]]

class MixtureModel {
  private:
    // MCMC parameters
    unsigned int _nBurn;     // number of burnin iterations
    unsigned int _nSample;   // number of posterior samples
    unsigned int _k;         // number of components
    unsigned int _s;         // current posterior iteration
    unsigned int _b;         // current burnin iteration

    // data
    arma::vec _data;
    unsigned int _n;

    // tuning parameters
    double _delta_theta;
    double _delta_sigma;

    // parameters
    arma::vec _theta;       // means
    arma::vec _sigma;       // variances
    arma::vec _lambda;      // proportions
    arma::ivec _z;    // latent assignment

    // stored chains
    arma::mat _theta_chain;       // means
    arma::mat _sigma_chain;       // variances
    arma::mat _lambda_chain;      // proportions

    void update_theta(bool save=false);
    void update_sigma(bool save=false);
    void update_lambda(bool save=false);
    void update_z();

  public:
    // constructor and destructor
    MixtureModel(arma::vec data, unsigned int k,
                 unsigned int burnin, unsigned int sample);
    MixtureModel(arma::vec data, unsigned int k,
                 arma::vec lambda, arma::ivec z,
                 unsigned int burnin, unsigned int sample);
    ~MixtureModel();

    // main methods
    void run_burnin();       // run burnin
    void posterior_sample(); // sample from stationary distribution
    Rcpp::List get_chains(); // return stored chains

};

#endif
