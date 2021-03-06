library(RcppMixtureModel)

theta <- c(5, 22, 40)
sigma <- c(2, 3, 1.5)
lambda <- c(0.3, 0.2, 0.5)
n <- 5000
y <- c(rnorm(n * lambda[1], theta[1], sqrt(sigma[1])),
       rnorm(n * lambda[2], theta[2], sqrt(sigma[2])),
       rnorm(n * lambda[3], theta[3], sqrt(sigma[3])))

system.time({
    out <- run.model(y, k=3, cores=1, burnin=5000, thin=2)
})

system.time({
    out <- run.model(y, k=3, cores=3, burnin=5000, thin=2)
})
