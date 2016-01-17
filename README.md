# RcppMixtureModel

Implementation of a Bayesian mixture model using Rcpp. Matrices and vectors are handled with Armadillo for thread safety. Open-MPI is used for running multiple MCMC chains so that convergence of parameters can be assessed using the Gelman-Rubin diagnostic.

Currently only Linux and OS X builds are supported. Linux compiler should work out of the box, but OS X will require using using gcc installed with `brew install gcc --without-multilib` instead of the default Apple provided clang
