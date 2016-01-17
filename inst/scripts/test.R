devtools::load_all()

theta <- c(5, 22, 40)
sigma <- c(2, 3, 1.5)
lambda <- c(0.3, 0.2, 0.5)
n <- 5000
y <- c(rnorm(n * lambda[1], theta[1], sqrt(sigma[1])),
       rnorm(n * lambda[2], theta[2], sqrt(sigma[2])),
       rnorm(n * lambda[3], theta[3], sqrt(sigma[3])))

system.time({
out <- run_model(sample(y), 3, 5, 5000, 1000)
})
summary(out$theta)
s <- 1 / out$sigma
summary(s)
z <- out$z
plot(density(colMeans(z)))
