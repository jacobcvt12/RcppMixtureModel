#ifndef _DISTRIBUTIONS_H_
#define _DISTRIBUTIONS_H_

#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

arma::vec rnorm(int n=1, double mean=0.0, double variance=1.0);
arma::vec rdirichlet(int n, arma::vec alpha);
std::vector<int> rmultinom(int n, arma::vec p);
std::vector<int> rz(unsigned int n, unsigned int k);

#endif
