#ifndef _MIXTUREMODEL_H_
#define _MIXTUREMODEL_H_

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

class MixtureModel {
  private:
    // MCMC parameters
    unsigned int _nBurn;     // number of burnin iterations
    unsigned int _nSample;   // number of posterior samples
    unsigned int _k;         // number of components

    // data
    arma::vec _data;

    // parameters
    arma::vec _theta;       // means
    arma::vec _sigma;       // variances
    arma::vec _lambda;      // proportions
    arma::vec _z;           // latent assignment

    void update_theta();
    void update_sigma();
    void update_lambda();
    void update_z();

  public:
    MixtureModel(arma::vec data, unsigned int k,
                 unsigned int burnin, unsigned int sample);

    ~MixtureModel();

    // main methods
    void run_burnin();       // run burnin
    void posterior_sample(); // sample from stationary distribution

};

#endif
