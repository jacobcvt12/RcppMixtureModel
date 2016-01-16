#ifndef _MIXTUREMODEL_H_
#define _MIXTUREMODEL_H_

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

class MixtureModel {
  private:
    unsigned int _nBurn;     // number of burnin iterations
    unsigned int _nSample;   // number of posterior samples
    unsigned int _k;         // number of components
    arma::vec _data;

    void update_mu();
    void update_lambda();
    void update_sigma();

  public:
    MixtureModel(arma::vec data, unsigned int k,
                 unsigned int burnin, unsigned int sample);

    ~MixtureModel();

    // main methods
    void run_burnin();       // run burnin
    void posterior_sample(); // sample from stationary distribution

};

#endif
