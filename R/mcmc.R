run.model <- function(data, k=2, thin=0, burnin=1000, sample=1000, cores=3) {
    if (k <= 0) 
        stop("k must be >= 1")
    if (cores <= 0)
        stop("cores must be >= 1")

    mcmc.out <- run_model(data, k, thin, burnin, sample, cores)

    return(mcmc.out)
}
