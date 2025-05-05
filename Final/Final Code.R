# Load required libraries
library(caret)
library(randomForest)
library(gbm)
library(e1071)
library(nnet)

set.seed(42)

##### (A) Load and Prepare Training Data
traindata <- read.table(file = "7406train.csv", sep = ",")
X1 <- traindata[, 1]
X2 <- traindata[, 2]
muhat <- apply(traindata[, 3:202], 1, mean)
Vhat  <- apply(traindata[, 3:202], 1, var)

# Apply log transformation to muhat and Vhat
log_muhat <- log(muhat)
log_Vhat  <- log(Vhat)

# Add log-transformed columns for training
data0 <- data.frame(X1 = X1, X2 = X2, muhat = muhat, Vhat = Vhat, log_muhat = log_muhat, log_Vhat = log_Vhat)

## we can plot 4 graphs in a single plot
par(mfrow = c(2, 2))
plot(X1, muhat)
plot(X2, muhat)
plot(X1, Vhat)
plot(X2, Vhat)

hist(data0$muhat)
boxplot(data0$muhat, main="Boxplot of Muhat", ylab="Values")
summary(data0$muhat)
hist(data0$Vhat)
summary(data0$Vhat)
boxplot(data0$Vhat, main="Boxplot of Vhat", ylab="Values")

##### (B) Cross-Validation Setup
cv_control <- trainControl(method = "cv", number = 10)

# Define tuning grids
rf_grid  <- expand.grid(mtry = c(1, 2))
svm_grid <- expand.grid(sigma = 0.01, C = c(1, 10))
gbm_grid <- expand.grid(
  n.trees = c(50, 100, 500),
  interaction.depth = c(1, 3),
  shrinkage = c(0.05, 0.1),
  n.minobsinnode = 10
)
nnet_grid <- expand.grid(size = c(2, 4), decay = c(0.1, 0.5))

##### (C) Train models for log(muhat)
models_logmu <- list()

models_logmu[["RF"]] <- train(
  log_muhat ~ X1 + X2, data = data0, method = "rf",
  trControl = cv_control, tuneGrid = rf_grid, ntree = 500
)

models_logmu[["GBM"]] <- train(
  log_muhat ~ X1 + X2, data = data0, method = "gbm",
  trControl = cv_control, tuneGrid = gbm_grid, verbose = FALSE
)

models_logmu[["SVM"]] <- train(
  log_muhat ~ X1 + X2, data = data0, method = "svmRadial",
  trControl = cv_control, tuneGrid = svm_grid
)

models_logmu[["NNET"]] <- train(
  log_muhat ~ X1 + X2, data = data0, method = "nnet",
  trControl = cv_control, tuneGrid = nnet_grid,
  linout = TRUE, trace = FALSE
)

# Select best log(muhat) model
results_logmu <- sapply(models_logmu, function(mod) min(mod$results$RMSE^2))
best_model_logmu <- names(which.min(results_logmu))
cat("Best model for log(muhat):", best_model_logmu, "\n")
models_logmu[[best_model_logmu]]$bestTune


##### (D) Train models for log(Vhat)
models_logvar <- list()

models_logvar[["RF"]] <- train(
  log_Vhat ~ X1 + X2, data = data0, method = "rf",
  trControl = cv_control, tuneGrid = rf_grid, ntree = 500
)

models_logvar[["GBM"]] <- train(
  log_Vhat ~ X1 + X2, data = data0, method = "gbm",
  trControl = cv_control, tuneGrid = gbm_grid, verbose = FALSE
)

models_logvar[["SVM"]] <- train(
  log_Vhat ~ X1 + X2, data = data0, method = "svmRadial",
  trControl = cv_control, tuneGrid = svm_grid
)

models_logvar[["NNET"]] <- train(
  log_Vhat ~ X1 + X2, data = data0, method = "nnet",
  trControl = cv_control, tuneGrid = nnet_grid,
  linout = TRUE, trace = FALSE
)

# Select best log(Vhat) model
results_logvar <- sapply(models_logvar, function(mod) min(mod$results$RMSE^2))
best_model_logvar <- names(which.min(results_logvar))
cat("Best model for log(Vhat):", best_model_logvar, "\n")
models_logvar[[best_model_logvar]]$bestTune


##### (E) Load Test Data and Predict
testX <- read.table(file = "7406test.csv", sep = ",")
colnames(testX) <- c("X1", "X2")

# Predict log(muhat)
logmu_pred <- predict(models_logmu[[best_model_logmu]], newdata = testX)
mu_pred <- exp(logmu_pred)  # Revert to original scale

# Predict log(Vhat) and revert to original scale
log_var_pred <- predict(models_logvar[[best_model_logvar]], newdata = testX)
var_pred <- exp(log_var_pred)

cat("Results for models predicting log(muhat):\n")
for (model_name in names(models_logmu)) {
  cat("\n---", model_name, "---\n")
  print(models_logmu[[model_name]]$results)
  cat("Best tuning parameters:\n")
  print(models_logmu[[model_name]]$bestTune)
  cat("Lowest RMSE:\n")
  print(min(models_logmu[[model_name]]$results$RMSE))
}
logmu_summary <- sapply(models_logmu, function(m) min(m$results$RMSE))
print(logmu_summary)

cat("\n\nResults for models predicting log(Vhat):\n")
for (model_name in names(models_logvar)) {
  cat("\n---", model_name, "---\n")
  print(models_logvar[[model_name]]$results)
  cat("Best tuning parameters:\n")
  print(models_logvar[[model_name]]$bestTune)
  cat("Lowest RMSE:\n")
  print(min(models_logvar[[model_name]]$results$RMSE))
}

logvar_summary <- sapply(models_logvar, function(m) min(m$results$RMSE))
print(logvar_summary)


##### (F) Finalize Submission Data and Write CSV
testdata <- data.frame(
  X1 = testX$X1,
  X2 = testX$X2,
  muhat = round(mu_pred, 6),
  Vhat  = round(var_pred, 6)
)



hist(testdata$muhat)
summary(testdata$muhat)
hist(testdata$Vhat)
summary(testdata$Vhat)


write.table(testdata, file="1.Malcolm.Olan.csv",
            sep=",",  col.names=F, row.names=F)
