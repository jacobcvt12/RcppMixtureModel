#ifndef _DISTRIBUTIONS_H_
#define _DISTRIBUTIONS_H_

#include <RcppArmadillo.h>
#include <vector>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

arma::vec rnorm(int n=1, double mean=0.0, double variance=1.0);
double dnorm(arma::vec y, double mean=0.0, double variance=1.0);
double dnorm(double y, double mean=0.0, double variance=1.0);
arma::vec rdirichlet(int n, arma::vec alpha);
arma::ivec rmultinom(int n, arma::vec p);
arma::ivec rz(unsigned int n, unsigned int k);

#endif
