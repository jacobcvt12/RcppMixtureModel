#ifndef _DISTRIBUTIONS_H_
#define _DISTRIBUTIONS_H_

#include <RcppArmadillo.h>
#include <vector>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

// normal random variables
arma::vec rnorm_cpp(int n=1, double mean=0.0, double variance=1.0);
double dnorm_cpp_vec(arma::vec y, double mean=0.0, double variance=1.0);
double dnorm_cpp(double y, double mean=0.0, double variance=1.0);

// gamma random variables
double dgamma_cpp(double y, double shape, double scale);

// other RVs
arma::vec rdirichlet_cpp(arma::vec alpha);
arma::ivec rmultinom_cpp(int n, arma::vec p);
arma::ivec rz_cpp(unsigned int n, unsigned int k);

#endif
