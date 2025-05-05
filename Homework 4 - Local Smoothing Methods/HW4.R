#PART 1 __________________________

set.seed(123)
library(DataExplorer)

## True function (f(x)) definition (eq. (2) as given)
f_true <- function(x) (1 - x^2) * exp(-0.5 * x^2)

## Number of simulations (Monte Carlo runs)
m <- 1000
n <- 101
x <- 2 * pi * seq(-1, 1, length=n)

## Initialize the matrix of fitted values for three methods
fvlp <- fvnw <- fvss <- matrix(0, nrow=n, ncol=m)


length(x)
summary(x)
plot_histogram(x)
plot_qq(x)


## Run the simulations
for (j in 1:m) {
  ## Simulate y-values based on true function and noise
  y <- (1 - x^2) * exp(-0.5 * x^2) + rnorm(length(x), sd=0.2)
  
  ## Get the estimates and store them
  fvlp[, j] <- predict(loess(y ~ x, span = 0.75), newdata = x)
  fvnw[, j] <- ksmooth(x, y, kernel="normal", bandwidth=0.2, x.points=x)$y
  fvss[, j] <- predict(smooth.spline(y ~ x), x=x)$y
}

## Initialize vectors to store the empirical bias, variance, and MSE for each method
bias_lp <- variance_lp <- mse_lp <- numeric(n)
bias_nw <- variance_nw <- mse_nw <- numeric(n)
bias_ss <- variance_ss <- mse_ss <- numeric(n)

## Compute bias, variance, and MSE for each method at each x_i
for (i in 1:n) {
  ## Collect the fitted values for the current x_i across all simulations
  y_lp <- fvlp[i, ]
  y_nw <- fvnw[i, ]
  y_ss <- fvss[i, ]
  
  ## True value at x_i
  true_value <- f_true(x[i])
  
  ## Compute the mean of the fitted values
  mean_lp <- mean(y_lp)
  mean_nw <- mean(y_nw)
  mean_ss <- mean(y_ss)
  
  ## Compute Bias (average of estimates - true value)
  bias_lp[i] <- mean_lp - true_value
  bias_nw[i] <- mean_nw - true_value
  bias_ss[i] <- mean_ss - true_value
  
  ## Compute Variance (variance of estimates)
  variance_lp[i] <- mean((y_lp - mean_lp)^2)
  variance_nw[i] <- mean((y_nw - mean_nw)^2)
  variance_ss[i] <- mean((y_ss - mean_ss)^2)
  
  ## Compute MSE (mean squared error between estimates and true value)
  mse_lp[i] <- mean((y_lp - true_value)^2)
  mse_nw[i] <- mean((y_nw - true_value)^2)
  mse_ss[i] <- mean((y_ss - true_value)^2)
}

## Below is the sample R code to plot the mean of three estimators in a single plot
meanlp = apply(fvlp,1,mean);
meannw = apply(fvnw,1,mean);
meanss = apply(fvss,1,mean);
dmin = min( meanlp,  meannw,  meanss);
dmax = max( meanlp,  meannw,  meanss);
matplot(x, meanlp, "l", ylim=c(dmin, dmax), ylab="Response")
matlines(x, meannw, col="red")
matlines(x, meanss, col="blue")
legend("topright", legend=c("LOESS", "Kernel", "Spline"), col=c("black", "red", "blue"), lty=1)
## You might add the raw observations to compare with the fitted curves
points(x,y)

## Plot the empirical bias, variance, and MSE

par(mar=c(5, 5, 4, 6))

## Plot Bias
plot(x, bias_lp, type="l", col="black", ylim=c(min(bias_lp, bias_nw, bias_ss), max(bias_lp, bias_nw, bias_ss)),
     ylab="Bias", xlab="x", main="Bias for Different Methods")
lines(x, bias_nw, col="red")
lines(x, bias_ss, col="blue")
legend("topright", legend=c("LOESS", "Kernel", "Spline"), col=c("black", "red", "blue"), lty=1)

## Plot Variance
plot(x, variance_lp, type="l", col="black", ylim=c(min(variance_lp, variance_nw, variance_ss), max(variance_lp, variance_nw, variance_ss)),
     ylab="Variance", xlab="x", main="Variance for Different Methods")
lines(x, variance_nw, col="red")
lines(x, variance_ss, col="blue")
legend("topright", legend=c("LOESS", "Kernel", "Spline"), col=c("black", "red", "blue"), lty=1)

## Plot MSE
plot(x, mse_lp, type="l", col="black", ylim=c(min(mse_lp, mse_nw, mse_ss), max(mse_lp, mse_nw, mse_ss)),
     ylab="MSE", xlab="x", main="MSE for Different Methods")
lines(x, mse_nw, col="red")
lines(x, mse_ss, col="blue")
legend("topright", legend=c("LOESS", "Kernel", "Spline"), col=c("black", "red", "blue"), lty=1)

# Now compute summary statistics for each method
bias_lp_mean <- mean(bias_lp)
var_lp_mean <- mean(variance_lp)
mse_lp_mean <- mean(mse_lp)

bias_nw_mean <- mean(bias_nw)
var_nw_mean <- mean(variance_nw)
mse_nw_mean <- mean(mse_nw)

bias_ss_mean <- mean(bias_ss)
var_ss_mean <- mean(variance_ss)
mse_ss_mean <- mean(mse_ss)

# Create a data frame to display the results in a readable format
summary_stats <- data.frame(
  Method = c("LOESS", "Kernel Smoothing", "Spline Smoothing"),
  Bias = c(bias_lp_mean, bias_nw_mean, bias_ss_mean),
  Variance = c(var_lp_mean, var_nw_mean, var_ss_mean),
  MSE = c(mse_lp_mean, mse_nw_mean, mse_ss_mean)
)

# Print the summary statistics
print(summary_stats)


#PART 2 ___________________________________________________________________

set.seed(79)

## True function (f(x)) definition (eq. (2) as given)
f_true2 <- function(x) (1 - x2^2) * exp(-0.5 * x2^2)

## Number of simulations (Monte Carlo runs)
m <- 1000
n <- 101
x2 <- round(2*pi*sort(c(0.5, -1 + rbeta(50,2,2), rbeta(50,2,2))), 8)

length(x2)
summary(x2)
plot_histogram(x2)
plot_qq(x2)


## Initialize the matrix of fitted values for three methods
fvlp2 <- fvnw2 <- fvss2 <- matrix(0, nrow=n, ncol=m)

## Run the simulations
for (j in 1:m) {
  ## Simulate y-values based on true function and noise
  y <- (1-x2^2) * exp(-0.5 * x2^2) + rnorm(length(x2), sd=0.2);
  fvlp2[,j] <- predict(loess(y ~ x2, span = 0.3365), newdata = x2);
  fvnw2[,j] <- ksmooth(x2, y, kernel="normal", bandwidth= 0.2, x.points=x2)$y;
 fvss2[,j] <-  predict(smooth.spline(y ~ x2, spar= 0.7163), x=x2)$y
}

## Initialize vectors to store the empirical bias, variance, and MSE for each method
bias_lp2 <- variance_lp2 <- mse_lp2 <- numeric(n)
bias_nw2 <- variance_nw2 <- mse_nw2 <- numeric(n)
bias_ss2 <- variance_ss2 <- mse_ss2 <- numeric(n)

## Compute bias, variance, and MSE for each method at each x_i
for (i in 1:n) {
  ## Collect the fitted values for the current x_i across all simulations
  y_lp2 <- fvlp2[i, ]
  y_nw2 <- fvnw2[i, ]
  y_ss2 <- fvss2[i, ]
  
  ## True value at x_i
  true_value2 <- f_true2(x[i])
  
  ## Compute the mean of the fitted values (fm(xi))
  mean_lp2 <- mean(y_lp2)
  mean_nw2 <- mean(y_nw2)
  mean_ss2 <- mean(y_ss2)
  
  ## Compute Bias (average of estimates - true value)
  bias_lp2[i] <- mean_lp2 - true_value2
  bias_nw2[i] <- mean_nw2 - true_value2
  bias_ss2[i] <- mean_ss2 - true_value2
  
  ## Compute Variance (variance of estimates)
  variance_lp2[i] <- mean((y_lp2 - mean_lp2)^2)
  variance_nw2[i] <- mean((y_nw2 - mean_nw2)^2)
  variance_ss2[i] <- mean((y_ss2 - mean_ss2)^2)
  
  ## Compute MSE (mean squared error between estimates and true value)
  mse_lp2[i] <- mean((y_lp2 - true_value2)^2)
  mse_nw2[i] <- mean((y_nw2 - true_value2)^2)
  mse_ss2[i] <- mean((y_ss2 - true_value2)^2)
}

## Below is the sample R code to plot the mean of three estimators in a single plot
meanlp2 = apply(fvlp2,1,mean);
meannw2 = apply(fvnw2,1,mean);
meanss2 = apply(fvss2,1,mean);
dmin2 = min( meanlp2,  meannw2,  meanss2);
dmax2 = max( meanlp2,  meannw2,  meanss2);
matplot(x2, meanlp2, "l", ylim=c(dmin2, dmax2), ylab="Response")
matlines(x2, meannw2, col="red")
matlines(x2, meanss2, col="blue")
legend("topright", legend=c("LOESS", "Kernel", "Spline"), col=c("black", "red", "blue"), lty=1)

## You might add the raw observations to compare with the fitted curves
points(x,y)

## Plot the empirical bias, variance, and MSE

par(mar=c(5, 5, 4, 6)) 

## Plot Bias
plot(x2, bias_lp2, type="l", col="black", ylim=c(min(bias_lp2, bias_nw2, bias_ss2), max(bias_lp2, bias_nw2, bias_ss2)),
     ylab="Bias", xlab="x2", main="Bias for Different Methods")
lines(x2, bias_nw2, col="red")
lines(x2, bias_ss2, col="blue")
legend("topright", legend=c("LOESS", "Kernel", "Spline"), col=c("black", "red", "blue"), lty=1)

## Plot Variance
plot(x2, variance_lp2, type="l", col="black", ylim=c(min(variance_lp2, variance_nw2, variance_ss2), max(variance_lp2, variance_nw2, variance_ss2)),
     ylab="Variance", xlab="x2", main="Variance for Different Methods")
lines(x2, variance_nw2, col="red")
lines(x2, variance_ss2, col="blue")
legend("topright", legend=c("LOESS", "Kernel", "Spline"), col=c("black", "red", "blue"), lty=1)

## Plot MSE
plot(x2, mse_lp2, type="l", col="black", ylim=c(min(mse_lp2, mse_nw2, mse_ss2), max(mse_lp2, mse_nw2, mse_ss2)),
     ylab="MSE", xlab="x2", main="MSE for Different Methods")
lines(x2, mse_nw2, col="red")
lines(x2, mse_ss2, col="blue")
legend("topright", legend=c("LOESS", "Kernel", "Spline"), col=c("black", "red", "blue"), lty=1)

# Now compute summary statistics for each method
bias_lp_mean2 <- mean(bias_lp2)
var_lp_mean2 <- mean(variance_lp2)
mse_lp_mean2 <- mean(mse_lp2)

bias_nw_mean2 <- mean(bias_nw2)
var_nw_mean2 <- mean(variance_nw2)
mse_nw_mean2 <- mean(mse_nw2)

bias_ss_mean2 <- mean(bias_ss2)
var_ss_mean2 <- mean(variance_ss2)
mse_ss_mean2 <- mean(mse_ss2)

# Create a data frame to display the results in a readable format
summary_stats2 <- data.frame(
  Method = c("LOESS", "Kernel Smoothing", "Spline Smoothing"),
  Bias = c(bias_lp_mean2, bias_nw_mean2, bias_ss_mean2),
  Variance = c(var_lp_mean2, var_nw_mean2, var_ss_mean2),
  MSE = c(mse_lp_mean2, mse_nw_mean2, mse_ss_mean2)
)

# Print the summary statistics
print(summary_stats2)
