// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// run_model
Rcpp::List run_model(arma::vec data, unsigned int k, unsigned int thin, unsigned int burnin, unsigned int sample, unsigned int cores);
RcppExport SEXP RcppMixtureModel_run_model(SEXP dataSEXP, SEXP kSEXP, SEXP thinSEXP, SEXP burninSEXP, SEXP sampleSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::vec >::type data(dataSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type sample(sampleSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type cores(coresSEXP);
    __result = Rcpp::wrap(run_model(data, k, thin, burnin, sample, cores));
    return __result;
END_RCPP
}
