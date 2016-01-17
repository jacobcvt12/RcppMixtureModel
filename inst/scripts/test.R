devtools::load_all()

theta <- c(5, 15)
sigma <- c(2, 3)
lambda <- c(0.3, 0.7)
n <- 1000
y <- c(rnorm(n * lambda[1], theta[1], sqrt(sigma[1])),
       rnorm(n * lambda[2], theta[2], sqrt(sigma[2])))
z <- c(rep(0, n * lambda[1]),
       rep(1, n * lambda[2]))

out <- run_model(y, 2, lambda, z, 10000, 5000)
summary(out$theta)
s <- 1 / out$sigma
summary(s)
