# RcppMixtureModel

Implementation of a Bayesian mixture model using Rcpp. Matrices and vectors are handled with Armadillo for thread safety. Open-MPI is used for running multiple MCMC chains so that convergence of parameters can be assessed using the Gelman-Rubin diagnostic.
