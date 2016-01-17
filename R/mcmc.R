run.model <- function(data, k=2, thin=0, burnin=1000, sample=1000, cores=3) {
    if (k <= 0) 
        stop("k must be >= 1")
    if (cores <= 0)
        stop("cores must be >= 1")

    mcmc.out <- run_model(data, k, thin, burnin, sample, cores)

    # calculate summary statistics
    theta <- mcmc.out$theta
    sigma <- mcmc.out$sigma

    # matrix for "pretty printing"
    mcmc.summary <- matrix(0, nrow=(k * 2), ncol=4)
    colnames(mcmc.summary) <- c("2.5%", "Mean", "97.5%", "R.hat")
    rownames(mcmc.summary) <- c(paste("theta", 1:3), paste("sigma", 1:3))

    for (i in 1:k) {
        # concatenate chains
        theta.all <- as.vector(theta[, i, ])
        sigma.all <- 1 / as.vector(sigma[, i, ])

        if (cores > 1) {
            # calculate diagnostic
            theta.rhat <- round(gelman.rubin(theta[, i, ]), 2)
            sigma.rhat <- round(gelman.rubin(sigma[, i, ]), 2)
        } else {
            theta.rhat <- NA
            sigma.rhat <- NA
        }

        # calculate mean summaries
        theta.mean <- mean(theta.all)
        theta.2.5 <- quantile(theta.all, 0.025)
        theta.97.5 <- quantile(theta.all, 0.975)

        # calculate variance summaries
        sigma.mean <- mean(sigma.all)
        sigma.2.5 <- quantile(sigma.all, 0.025)
        sigma.97.5 <- quantile(sigma.all, 0.975)

        # store in matrix
        mcmc.summary[paste("theta", i), ] <- c(theta.2.5, theta.mean,
                                               theta.97.5, theta.rhat)
        mcmc.summary[paste("sigma", i), ] <- c(sigma.2.5, sigma.mean,
                                               sigma.97.5, sigma.rhat)
    }

    print(mcmc.summary)

    return(mcmc.out)
}

gelman.rubin <- function(param) {
    # mcmc information
    n <- nrow(param) # number of iterations
    m <- ncol(param) # number of chains

    # calculate the mean of the means
    theta.bar.bar <- mean(colMeans(param))

    # within chain variance
    W <- mean(apply(param, 2, var))

    # between chain variance
    B <- n / (m - 1) * sum((colMeans(param) - theta.bar.bar) ^ 2)

    # variance of stationary distribution
    theta.var.hat <- (1 - 1 / n) * W + 1 / n * B

    # Potential Scale Reduction Factor (PSRF)
    R.hat <- sqrt(theta.var.hat / W)

    return(R.hat)
}
