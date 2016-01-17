devtools::load_all()

theta <- c(5, 13, 15)
sigma <- c(2, 3, 1.5)
lambda <- c(0.3, 0.2, 0.5)
n <- 10000
y <- c(rnorm(n * lambda[1], theta[1], sqrt(sigma[1])),
       rnorm(n * lambda[2], theta[2], sqrt(sigma[2])),
       rnorm(n * lambda[3], theta[3], sqrt(sigma[3])))
z <- c(rep(0, n * lambda[1]),
       rep(1, n * lambda[2]),
       rep(2, n * lambda[3]))

system.time({
out <- run_model(y, 3, z, 10000, 5000)
})
summary(out$theta)
s <- 1 / out$sigma
summary(s)
