data <- read.csv("stroke_risk_dataset.csv")
head(data)

#convert categorical variables to factors
data[, 1:15] <- lapply(data[, 1:15], factor)
#rename columns
names(data) <- c('Chest.Pain', 'Shortness.of.Breath', 'Irregular.Heartbeat', 'Fatigue.Weakness', 'Dizziness', 'Swelling.Edema', 'Pain.in.Neck.Jaw.Shoulder.Back', 'Excessive.Sweating', 'Persistent.Cough', 'Nausea.Vomiting', 'High.Blood.Pressure', 'Chest.Discomfort.Activity', 'Cold.Hands.Feet', 'Snoring.Sleep.Apnea', 'Anxiety.Feeling.of.Doom', 'Age', 'Stroke.Risk.Percentage', 'At.Risk')

#Data Prep
library(DataExplorer)
plot_histogram(data)
boxplot(data, main = "Boxplots of All Variables")
boxplot(data['Stroke.Risk.Percentage'])
summary(data)

sum(data$At.Risk == 0)
sum(data$At.Risk == 1)

#EDA
library(corrplot)
introduce(data)
plot_bar(data)
df_numeric <- as.data.frame(lapply(data, function(x) as.numeric(as.character(x))))
correlation_matrix <- cor(df_numeric)
corrplot(correlation_matrix)


#split data into training and testing
set.seed(123)

library(dplyr)
# Undersample the majority class (At.Risk == 1)
df_balanced <- data %>%
  group_by(At.Risk) %>%
  sample_n(24556) %>%  # Match the minority class count
  ungroup()

library(ggplot2)
library(dplyr)
library(gridExtra)

# Example datasets
df1 <- data.frame(data)
df2 <- data.frame(df_balanced)

# Count occurrences of At.Risk values in each dataset
df1_counts <- df1 %>%
  count(At.Risk) %>%
  mutate(Dataset = "Dataset 1",
         At.Risk = factor(At.Risk, levels = c(0, 1)))  # Convert to factor

df2_counts <- df2 %>%
  count(At.Risk) %>%
  mutate(Dataset = "Dataset 2",
         At.Risk = factor(At.Risk, levels = c(0, 1)))  # Convert to factor

# Plot for Dataset 1
plot1 <- ggplot(df1_counts, aes(x = At.Risk, y = n, fill = At.Risk)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  geom_text(aes(label = n), vjust = 2, size = 5) +
  labs(title = "Dataset 1 - At Risk Counts",
       x = "At.Risk Value", y = "Count") +
  theme_minimal()

# Plot for Dataset 2
plot2 <- ggplot(df2_counts, aes(x = At.Risk, y = n, fill = At.Risk)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  geom_text(aes(label = n), vjust = 2, size = 5) +
  labs(title = "Dataset 2 - At Risk Counts",
       x = "At.Risk Value", y = "Count") +
  theme_minimal()

# Display both plots side by side
grid.arrange(plot1, plot2, ncol = 2)

flag <- floor(0.70 * nrow(df_balanced))
train_ind <- sample(seq_len(nrow(df_balanced)), size = flag)
train <- df_balanced[train_ind, ]
test <- df_balanced[-train_ind, ]


#regression tree to determine age ranges_____________________________________________________________
library(rpart)
library(rpart.plot)
treetraindata <- subset(train, select=-c(At.Risk))
treetestdata <- subset(test, select = -c(At.Risk))

#trianing and testing data
tree <- rpart(Stroke.Risk.Percentage ~., data = treetraindata, method = 'anova')
summary(tree)
rpart.plot(tree, type=2, extra=101, fallen.leaves=TRUE, cex=0.8, main="Decision Tree for Stroke Risk")

opt_index <- which.min(tree$cptable[, "xerror"])  # Minimum xerror
se_rule <- tree$cptable[opt_index, "xerror"] + tree$cptable[opt_index, "xstd"]  # 1-SE Rule
cp_best <- max(tree$cptable[tree$cptable[, "xerror"] <= se_rule, "CP"])  # Simplest CP within 1-SE

pruned_tree <- prune(tree, cp = cp_best)

summary(pruned_tree)
rpart.plot(pruned_tree, type=2, extra=101, fallen.leaves=TRUE, cex=0.8, main="Pruned Decision Tree for Stroke Risk")

# Predictions on training data
train_preds <- predict(pruned_tree, treetraindata)

# Compute Mean Squared Error (MSE) for training data
mse_train <- mean((treetraindata$Stroke.Risk.Percentage - train_preds)^2)
cat("MSE (Training):", mse_train, "\n")

sst <- var(treetraindata$Stroke.Risk.Percentage) * nrow(treetraindata)  # Total Sum of Squares
r2_train <- 1 - (mse_train / (sst / nrow(treetraindata)))
cat("R² (Training):", r2_train, "\n")


# Predictions on test data
test_preds <- predict(pruned_tree, treetestdata)

# Compute Mean Squared Error (MSE) for test data
mse_test <- mean((treetestdata$Stroke.Risk.Percentage - test_preds)^2)
cat("MSE (Test):", mse_test, "\n")

sst_test <- var(treetestdata$Stroke.Risk.Percentage) * nrow(treetestdata)
r2_test <- 1 - (mse_test / (sst_test / nrow(treetestdata)))
cat("R² (Test):", r2_test, "\n")

#cross validation for reg tree_______________________________________________________________________
library(caret)
# ----------------------------
# 1. Cross-validation to find best cp
# ----------------------------

# Set up cross-validation and grid for cp tuning
train_control <- trainControl(method = "cv", number = 10)
grid <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))

# Perform cross-validation
cv_tree_model <- train(Stroke.Risk.Percentage ~ ., 
                       data = treetraindata, 
                       method = "rpart", 
                       trControl = train_control,
                       tuneGrid = grid)

# Extract best cp
best_cp <- cv_tree_model$bestTune$cp
cat("Best cp found:", best_cp, "\n")

# ----------------------------
# 2. Refit the final model with best cp on full training data
# ----------------------------

final_tree_model <- rpart(Stroke.Risk.Percentage ~ ., 
                          data = treetraindata, 
                          cp = best_cp)

# Plot final tree
rpart.plot(final_tree_model, type = 2, extra = 101, fallen.leaves = TRUE, 
           cex = 0.8, main = "Final Pruned CV Decision Tree")
summary(final_tree_model)
# ----------------------------
# 3. Evaluate on Training Data
# ----------------------------

# Predictions on training set
train_preds <- predict(final_tree_model, treetraindata)

# Training MSE
mse_train <- mean((treetraindata$Stroke.Risk.Percentage - train_preds)^2)

# Training R²
sst_train <- var(treetraindata$Stroke.Risk.Percentage) * nrow(treetraindata)
r2_train <- 1 - (mse_train / (sst_train / nrow(treetraindata)))

cat("Training MSE:", mse_train, "\n")
cat("Training R²:", r2_train, "\n")

# ----------------------------
# 4. Evaluate on Test Data
# ----------------------------

# Predictions on test set
test_preds <- predict(final_tree_model, treetestdata)

# Test MSE
mse_test <- mean((treetestdata$Stroke.Risk.Percentage - test_preds)^2)

# Test R²
sst_test <- var(treetestdata$Stroke.Risk.Percentage) * nrow(treetestdata)
r2_test <- 1 - (mse_test / (sst_test / nrow(treetestdata)))

cat("Test MSE:", mse_test, "\n")
cat("Test R²:", r2_test, "\n")

#Random Forest________________________________________________________________________________________
library(randomForest)
forest <- randomForest(Stroke.Risk.Percentage ~ ., data = treetraindata, mtry = floor(sqrt(ncol(treetraindata))),
                       ntree = 100, importance = TRUE)
importance(forest)
varImpPlot(forest)

# Predict on training set
predicted_train <- predict(forest, treetraindata)
actual_train <- treetraindata$Stroke.Risk.Percentage

# Compute MSE and R² on training set
mse_train_forest <- mean((predicted_train - actual_train)^2)
r_squared_train_forest <- 1 - sum((predicted_train - actual_train)^2) / sum((actual_train - mean(actual_train))^2)

cat("Training MSE (Random Forest):", mse_train_forest, "\n")
cat("Training R² (Random Forest):", r_squared_train_forest, "\n")

predicted <- predict(forest, treetestdata)
actual <- treetestdata$Stroke.Risk.Percentage
mse_forest <- mean((predicted - actual)^2)
r_squared_forest <- 1 - sum((predicted - actual)^2) / sum((actual - mean(actual))^2)
print(mse_forest)
print(r_squared_forest)

#cross validation on RF________________________________________________________________________________
mtry_grid <- expand.grid(mtry = seq(1, floor(sqrt(ncol(treetraindata))), by = 1))
cv_rf <- train(Stroke.Risk.Percentage ~ ., 
               data = treetraindata, 
               method = "rf", 
               trControl = train_control,
               tuneGrid = mtry_grid,
               ntree = 100,  
               importance = TRUE)

print(cv_rf)  # Shows best mtry and RMSE
plot(cv_rf)  # Visual of RMSE vs mtry

# Best mtry value
cv_rf$bestTune

# Retrain final model with best mtry on full training data
final_rf <- randomForest(Stroke.Risk.Percentage ~ ., 
                         data = treetraindata,
                         mtry = cv_rf$bestTune$mtry,
                         ntree = 100, 
                         importance = TRUE)

importance(final_rf)
varImpPlot(final_rf)

# Training predictions
train_preds_final <- predict(final_rf, treetraindata)

# Training MSE
mse_train_final <- mean((train_preds_final - treetraindata$Stroke.Risk.Percentage)^2)

# Training R²
sst_train <- var(treetraindata$Stroke.Risk.Percentage) * nrow(treetraindata)
r2_train_final <- 1 - (mse_train_final / (sst_train / nrow(treetraindata)))

cat("Final MSE (Training):", mse_train_final, "\n")
cat("Final R² (Training):", r2_train_final, "\n")


# Test predictions
test_preds_final <- predict(final_rf, treetestdata)

# Test MSE
mse_test_final <- mean((test_preds_final - treetestdata$Stroke.Risk.Percentage)^2)

# Test R²
sst_test <- var(treetestdata$Stroke.Risk.Percentage) * nrow(treetestdata)
r2_test_final <- 1 - (mse_test_final / (sst_test / nrow(treetestdata)))

cat("Final MSE (Test):", mse_test_final, "\n")
cat("Final R² (Test):", r2_test_final, "\n")

##WITHOUT AGE______________________________________________________________########################
#regression tree to determine age ranges_____________________________________________________________
library(rpart)
library(rpart.plot)
treetraindata2 <- subset(train, select=-c(At.Risk, Age))
treetestdata2 <- subset(test, select = -c(At.Risk, Age))

#trianing and testing data
tree2 <- rpart(Stroke.Risk.Percentage ~., data = treetraindata2, method = 'anova')
summary(tree2)
rpart.plot(tree2, type=2, extra=101, fallen.leaves=TRUE, cex=0.8, main="Decision Tree for Stroke Risk")

opt_index2 <- which.min(tree2$cptable[, "xerror"])  # Minimum xerror
se_rule2 <- tree2$cptable[opt_index2, "xerror"] + tree2$cptable[opt_index2, "xstd"]  # 1-SE Rule
cp_best2 <- max(tree2$cptable[tree2$cptable[, "xerror"] <= se_rule2, "CP"])  # Simplest CP within 1-SE

pruned_tree2 <- prune(tree2, cp = cp_best2)

summary(pruned_tree2)
rpart.plot(pruned_tree2, type=2, extra=101, fallen.leaves=TRUE, cex=0.8, main="Pruned Decision Tree for Stroke Risk")

# Predictions on training data
train_preds2 <- predict(pruned_tree2, treetraindata2)

# Compute Mean Squared Error (MSE) for training data
mse_train2 <- mean((treetraindata2$Stroke.Risk.Percentage - train_preds2)^2)
cat("MSE (Training):", mse_train2, "\n")

sst2 <- var(treetraindata2$Stroke.Risk.Percentage) * nrow(treetraindata2)  # Total Sum of Squares
r2_train2 <- 1 - (mse_train2 / (sst2 / nrow(treetraindata2)))
cat("R² (Training):", r2_train2, "\n")


# Predictions on test data
test_preds2 <- predict(pruned_tree2, treetestdata2)

# Compute Mean Squared Error (MSE) for test data
mse_test2 <- mean((treetestdata2$Stroke.Risk.Percentage - test_preds2)^2)
cat("MSE (Test):", mse_test2, "\n")

sst_test2 <- var(treetestdata2$Stroke.Risk.Percentage) * nrow(treetestdata2)
r2_test2 <- 1 - (mse_test2 / (sst_test2 / nrow(treetestdata2)))
cat("R² (Test):", r2_test2, "\n")

#cross validation for reg tree_______________________________________________________________________
library(caret)
# ----------------------------
# 1. Cross-validation to find best cp
# ----------------------------

# Set up cross-validation and grid for cp tuning
train_control2 <- trainControl(method = "cv", number = 10)
grid2 <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))

# Perform cross-validation
cv_tree_model2 <- train(Stroke.Risk.Percentage ~ ., 
                       data = treetraindata2, 
                       method = "rpart", 
                       trControl = train_control2,
                       tuneGrid = grid2)

# Extract best cp
best_cp2 <- cv_tree_model2$bestTune$cp
cat("Best cp found:", best_cp2, "\n")

# ----------------------------
# 2. Refit the final model with best cp on full training data
# ----------------------------

final_tree_model2 <- rpart(Stroke.Risk.Percentage ~ ., 
                          data = treetraindata2, 
                          cp = best_cp2)

# Plot final tree
rpart.plot(final_tree_model2, type = 2, extra = 101, fallen.leaves = TRUE, 
           cex = 0.8, main = "Final Pruned CV Decision Tree")
summary(final_tree_model2)
# ----------------------------
# 3. Evaluate on Training Data
# ----------------------------

# Predictions on training set
train_preds2 <- predict(final_tree_model2, treetraindata2)

# Training MSE
mse_train2 <- mean((treetraindata2$Stroke.Risk.Percentage - train_preds2)^2)

# Training R²
sst_train2 <- var(treetraindata2$Stroke.Risk.Percentage) * nrow(treetraindata2)
r2_train2 <- 1 - (mse_train2 / (sst_train2 / nrow(treetraindata2)))

cat("Training MSE:", mse_train2, "\n")
cat("Training R²:", r2_train2, "\n")

# ----------------------------
# 4. Evaluate on Test Data
# ----------------------------

# Predictions on test set
test_preds2 <- predict(final_tree_model2, treetestdata2)

# Test MSE
mse_test2 <- mean((treetestdata2$Stroke.Risk.Percentage - test_preds2)^2)

# Test R²
sst_test2 <- var(treetestdata2$Stroke.Risk.Percentage) * nrow(treetestdata2)
r2_test2 <- 1 - (mse_test2 / (sst_test2 / nrow(treetestdata2)))

cat("Test MSE:", mse_test2, "\n")
cat("Test R²:", r2_test2, "\n")

#Random Forest________________________________________________________________________________________
library(randomForest)
forest2 <- randomForest(Stroke.Risk.Percentage ~ ., data = treetraindata2, mtry = floor(sqrt(ncol(treetraindata2))),
                       ntree = 100, importance = TRUE)
importance(forest2)
varImpPlot(forest2)

# Predict on training set
predicted_train2 <- predict(forest2, treetraindata2)
actual_train2 <- treetraindata2$Stroke.Risk.Percentage

# Compute MSE and R² on training set
mse_train_forest2 <- mean((predicted_train2 - actual_train2)^2)
r_squared_train_forest2 <- 1 - sum((predicted_train2 - actual_train2)^2) / sum((actual_train2 - mean(actual_train2))^2)

cat("Training MSE (Random Forest):", mse_train_forest2, "\n")
cat("Training R² (Random Forest):", r_squared_train_forest2, "\n")

predicted2 <- predict(forest2, treetestdata2)
actual2 <- treetestdata2$Stroke.Risk.Percentage
mse_forest2 <- mean((predicted2 - actual2)^2)
r_squared_forest2 <- 1 - sum((predicted2 - actual2)^2) / sum((actual2 - mean(actual2))^2)
print(mse_forest2)
print(r_squared_forest2)

#cross validation on RF________________________________________________________________________________
mtry_grid2 <- expand.grid(mtry = seq(1, floor(sqrt(ncol(treetraindata2))), by = 1))
cv_rf2 <- train(Stroke.Risk.Percentage ~ ., 
               data = treetraindata2, 
               method = "rf", 
               trControl = train_control2,
               tuneGrid = mtry_grid2,
               ntree = 100,  
               importance = TRUE)

print(cv_rf2)  # Shows best mtry and RMSE
plot(cv_rf2)  # Visual of RMSE vs mtry

# Best mtry value
cv_rf2$bestTune

# Retrain final model with best mtry on full training data
final_rf2 <- randomForest(Stroke.Risk.Percentage ~ ., 
                         data = treetraindata2,
                         mtry = cv_rf2$bestTune$mtry,
                         ntree = 100, 
                         importance = TRUE)

importance(final_rf2)
varImpPlot(final_rf2)

# Training predictions
train_preds_final2 <- predict(final_rf2, treetraindata2)

# Training MSE
mse_train_final2 <- mean((train_preds_final2 - treetraindata2$Stroke.Risk.Percentage)^2)

# Training R²
sst_train2 <- var(treetraindata2$Stroke.Risk.Percentage) * nrow(treetraindata2)
r2_train_final2 <- 1 - (mse_train_final2 / (sst_train2 / nrow(treetraindata2)))

cat("Final MSE (Training):", mse_train_final2, "\n")
cat("Final R² (Training):", r2_train_final2, "\n")


# Test predictions
test_preds_final2 <- predict(final_rf2, treetestdata2)

# Test MSE
mse_test_final2 <- mean((test_preds_final2 - treetestdata2$Stroke.Risk.Percentage)^2)

# Test R²
sst_test2 <- var(treetestdata2$Stroke.Risk.Percentage) * nrow(treetestdata2)
r2_test_final2 <- 1 - (mse_test_final2 / (sst_test2 / nrow(treetestdata2)))

cat("Final MSE (Test):", mse_test_final2, "\n")
cat("Final R² (Test):", r2_test_final2, "\n")

##WITH TOP 8______________________________________________________________########################
#regression tree to determine age ranges_____________________________________________________________
library(rpart)
library(rpart.plot)
treetraindata3 <- subset(train, select=c(Cold.Hands.Feet, Fatigue.Weakness, Chest.Pain, Excessive.Sweating, High.Blood.Pressure, Anxiety.Feeling.of.Doom, Nausea.Vomiting, Snoring.Sleep.Apnea, Stroke.Risk.Percentage))
treetestdata3 <- subset(test, select = c(Cold.Hands.Feet, Fatigue.Weakness, Chest.Pain, Excessive.Sweating, High.Blood.Pressure, Anxiety.Feeling.of.Doom, Nausea.Vomiting, Snoring.Sleep.Apnea, Stroke.Risk.Percentage))

#trianing and testing data
tree3 <- rpart(Stroke.Risk.Percentage ~., data = treetraindata3, method = 'anova')
summary(tree3)
rpart.plot(tree3, type=2, extra=101, fallen.leaves=TRUE, cex=0.8, main="Decision Tree for Stroke Risk")

opt_index3 <- which.min(tree3$cptable[, "xerror"])  # Minimum xerror
se_rule3 <- tree3$cptable[opt_index3, "xerror"] + tree3$cptable[opt_index3, "xstd"]  # 1-SE Rule
cp_best3 <- max(tree3$cptable[tree3$cptable[, "xerror"] <= se_rule3, "CP"])  # Simplest CP within 1-SE

pruned_tree3 <- prune(tree3, cp = cp_best3)

summary(pruned_tree3)
rpart.plot(pruned_tree3, type=2, extra=101, fallen.leaves=TRUE, cex=0.8, main="Pruned Decision Tree for Stroke Risk")

# Predictions on training data
train_preds3 <- predict(pruned_tree3, treetraindata3)

# Compute Mean Squared Error (MSE) for training data
mse_train3 <- mean((treetraindata3$Stroke.Risk.Percentage - train_preds3)^2)
cat("MSE (Training):", mse_train3, "\n")
print(sqrt(mse_train3))

sst3 <- var(treetraindata3$Stroke.Risk.Percentage) * nrow(treetraindata3)  # Total Sum of Squares
r2_train3 <- 1 - (mse_train3 / (sst3 / nrow(treetraindata3)))
cat("R² (Training):", r2_train3, "\n")


# Predictions on test data
test_preds3 <- predict(pruned_tree3, treetestdata3)

# Compute Mean Squared Error (MSE) for test data
mse_test3 <- mean((treetestdata3$Stroke.Risk.Percentage - test_preds3)^2)
cat("MSE (Test):", mse_test3, "\n")
print(sqrt(mse_test3))


sst_test3 <- var(treetestdata3$Stroke.Risk.Percentage) * nrow(treetestdata3)
r2_test3 <- 1 - (mse_test3 / (sst_test3 / nrow(treetestdata3)))
cat("R² (Test):", r2_test3, "\n")

#cross validation for reg tree_______________________________________________________________________
library(caret)
# ----------------------------
# 1. Cross-validation to find best cp
# ----------------------------

# Set up cross-validation and grid for cp tuning
train_control3 <- trainControl(method = "cv", number = 10)
grid3 <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))

# Perform cross-validation
cv_tree_model3 <- train(Stroke.Risk.Percentage ~ ., 
                        data = treetraindata3, 
                        method = "rpart", 
                        trControl = train_control3,
                        tuneGrid = grid3)

# Extract best cp
best_cp3 <- cv_tree_model3$bestTune$cp
cat("Best cp found:", best_cp3, "\n")

# ----------------------------
# 2. Refit the final model with best cp on full training data
# ----------------------------

final_tree_model3 <- rpart(Stroke.Risk.Percentage ~ ., 
                           data = treetraindata3, 
                           cp = best_cp3)

# Plot final tree
rpart.plot(final_tree_model3, type = 2, extra = 101, fallen.leaves = TRUE, 
           cex = 0.8, main = "Final Pruned CV Decision Tree")
summary(final_tree_model3)
# ----------------------------
# 3. Evaluate on Training Data
# ----------------------------

# Predictions on training set
train_preds3 <- predict(final_tree_model3, treetraindata3)

# Training MSE
mse_train3 <- mean((treetraindata3$Stroke.Risk.Percentage - train_preds3)^2)

# Training R²
sst_train3 <- var(treetraindata3$Stroke.Risk.Percentage) * nrow(treetraindata3)
r2_train3 <- 1 - (mse_train3 / (sst_train3 / nrow(treetraindata3)))

cat("Training MSE:", mse_train3, "\n")
print(sqrt(mse_train3))
cat("Training R²:", r2_train3, "\n")

# ----------------------------
# 4. Evaluate on Test Data
# ----------------------------

# Predictions on test set
test_preds3 <- predict(final_tree_model3, treetestdata3)

# Test MSE
mse_test3 <- mean((treetestdata3$Stroke.Risk.Percentage - test_preds3)^2)

# Test R²
sst_test3 <- var(treetestdata3$Stroke.Risk.Percentage) * nrow(treetestdata3)
r2_test3 <- 1 - (mse_test3 / (sst_test3 / nrow(treetestdata3)))

cat("Test MSE:", mse_test3, "\n")
print(sqrt(mse_test3))
cat("Test R²:", r2_test3, "\n")

#Random Forest________________________________________________________________________________________
library(randomForest)
forest3 <- randomForest(Stroke.Risk.Percentage ~ ., data = treetraindata3, mtry = floor(sqrt(ncol(treetraindata3))),
                        ntree = 100, importance = TRUE)
importance(forest3)
varImpPlot(forest3)

# Predict on training set
predicted_train3 <- predict(forest3, treetraindata3)
actual_train3 <- treetraindata3$Stroke.Risk.Percentage

# Compute MSE and R² on training set
mse_train_forest3 <- mean((predicted_train3 - actual_train3)^2)
r_squared_train_forest3 <- 1 - sum((predicted_train3 - actual_train3)^2) / sum((actual_train3 - mean(actual_train3))^2)

cat("Training MSE (Random Forest):", mse_train_forest3, "\n")
print(sqrt(mse_train_forest3))
cat("Training R² (Random Forest):", r_squared_train_forest3, "\n")

predicted3 <- predict(forest3, treetestdata3)
actual3 <- treetestdata3$Stroke.Risk.Percentage
mse_forest3 <- mean((predicted3 - actual3)^2)
r_squared_forest3 <- 1 - sum((predicted3 - actual3)^2) / sum((actual3 - mean(actual3))^2)
cat("Tresting MSE (Random Forest):", mse_forest3, "\n")
print(sqrt(mse_forest3))
cat("Testing R² (Random Forest):", r_squared_forest3, "\n")


#cross validation on RF________________________________________________________________________________
mtry_grid3 <- expand.grid(mtry = seq(1, floor(sqrt(ncol(treetraindata3))), by = 1))
cv_rf3 <- train(Stroke.Risk.Percentage ~ ., 
                data = treetraindata3, 
                method = "rf", 
                trControl = train_control3,
                tuneGrid = mtry_grid3,
                ntree = 100,  
                importance = TRUE)

print(cv_rf3)  # Shows best mtry and RMSE
plot(cv_rf3)  # Visual of RMSE vs mtry

# Best mtry value
cv_rf3$bestTune

# Retrain final model with best mtry on full training data
final_rf3 <- randomForest(Stroke.Risk.Percentage ~ ., 
                          data = treetraindata3,
                          mtry = cv_rf3$bestTune$mtry,
                          ntree = 100, 
                          importance = TRUE)

importance(final_rf3)
varImpPlot(final_rf3)

# Training predictions
train_preds_final3 <- predict(final_rf3, treetraindata3)

# Training MSE
mse_train_final3 <- mean((train_preds_final3 - treetraindata3$Stroke.Risk.Percentage)^2)

# Training R²
sst_train3 <- var(treetraindata3$Stroke.Risk.Percentage) * nrow(treetraindata3)
r2_train_final3 <- 1 - (mse_train_final3 / (sst_train3 / nrow(treetraindata3)))

cat("Final MSE (Training):", mse_train_final3, "\n")
print(sqrt(mse_train_final3))
cat("Final R² (Training):", r2_train_final3, "\n")


# Test predictions
test_preds_final3 <- predict(final_rf3, treetestdata3)

# Test MSE
mse_test_final3 <- mean((test_preds_final3 - treetestdata3$Stroke.Risk.Percentage)^2)

# Test R²
sst_test3 <- var(treetestdata3$Stroke.Risk.Percentage) * nrow(treetestdata3)
r2_test_final3 <- 1 - (mse_test_final3 / (sst_test3 / nrow(treetestdata3)))

cat("Final MSE (Test):", mse_test_final3, "\n")
print(sqrt(mse_test_final3))
cat("Final R² (Test):", r2_test_final3, "\n")

##WITH TOP 9______________________________________________________________########################
#regression tree to determine age ranges_____________________________________________________________
library(rpart)
library(rpart.plot)
treetraindata4 <- subset(train, select=c(Cold.Hands.Feet, Irregular.Heartbeat, Fatigue.Weakness, Chest.Pain, Excessive.Sweating, High.Blood.Pressure, Anxiety.Feeling.of.Doom, Nausea.Vomiting, Snoring.Sleep.Apnea, Stroke.Risk.Percentage))
treetestdata4 <- subset(test, select = c(Cold.Hands.Feet, Irregular.Heartbeat, Fatigue.Weakness, Chest.Pain, Excessive.Sweating, High.Blood.Pressure, Anxiety.Feeling.of.Doom, Nausea.Vomiting, Snoring.Sleep.Apnea, Stroke.Risk.Percentage))

#trianing and testing data
tree4 <- rpart(Stroke.Risk.Percentage ~., data = treetraindata4, method = 'anova')
summary(tree4)
rpart.plot(tree4, type=2, extra=101, fallen.leaves=TRUE, cex=0.8, main="Decision Tree for Stroke Risk")

opt_index4 <- which.min(tree4$cptable[, "xerror"])  # Minimum xerror
se_rule4 <- tree4$cptable[opt_index4, "xerror"] + tree4$cptable[opt_index4, "xstd"]  # 1-SE Rule
cp_best4 <- max(tree4$cptable[tree4$cptable[, "xerror"] <= se_rule4, "CP"])  # Simplest CP within 1-SE

pruned_tree4 <- prune(tree4, cp = cp_best4)

summary(pruned_tree4)
rpart.plot(pruned_tree4, type=2, extra=101, fallen.leaves=TRUE, cex=0.8, main="Pruned Decision Tree for Stroke Risk")

# Predictions on training data
train_preds4 <- predict(pruned_tree4, treetraindata4)

# Compute Mean Squared Error (MSE) for training data
mse_train4 <- mean((treetraindata4$Stroke.Risk.Percentage - train_preds4)^2)
cat("MSE (Training):", mse_train4, "\n")
print(sqrt(mse_train4))

sst4 <- var(treetraindata4$Stroke.Risk.Percentage) * nrow(treetraindata4)  # Total Sum of Squares
r2_train4 <- 1 - (mse_train4 / (sst4 / nrow(treetraindata4)))
cat("R² (Training):", r2_train4, "\n")


# Predictions on test data
test_preds4 <- predict(pruned_tree4, treetestdata4)

# Compute Mean Squared Error (MSE) for test data
mse_test4 <- mean((treetestdata4$Stroke.Risk.Percentage - test_preds4)^2)
cat("MSE (Test):", mse_test4, "\n")
print(sqrt(mse_test4))

sst_test4 <- var(treetestdata4$Stroke.Risk.Percentage) * nrow(treetestdata4)
r2_test4 <- 1 - (mse_test4 / (sst_test4 / nrow(treetestdata4)))
cat("R² (Test):", r2_test4, "\n")

#cross validation for reg tree_______________________________________________________________________
library(caret)
# ----------------------------
# 1. Cross-validation to find best cp
# ----------------------------

# Set up cross-validation and grid for cp tuning
train_control4 <- trainControl(method = "cv", number = 10)
grid4 <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))

# Perform cross-validation
cv_tree_model4 <- train(Stroke.Risk.Percentage ~ ., 
                        data = treetraindata4, 
                        method = "rpart", 
                        trControl = train_control4,
                        tuneGrid = grid4)

# Extract best cp
best_cp4 <- cv_tree_model4$bestTune$cp
cat("Best cp found:", best_cp4, "\n")

# ----------------------------
# 2. Refit the final model with best cp on full training data
# ----------------------------

final_tree_model4 <- rpart(Stroke.Risk.Percentage ~ ., 
                           data = treetraindata4, 
                           cp = best_cp4)

# Plot final tree
rpart.plot(final_tree_model4, type = 2, extra = 101, fallen.leaves = TRUE, 
           cex = 0.8, main = "Final Pruned CV Decision Tree")
summary(final_tree_model4)
# ----------------------------
# 3. Evaluate on Training Data
# ----------------------------

# Predictions on training set
train_preds4 <- predict(final_tree_model4, treetraindata4)

# Training MSE
mse_train4 <- mean((treetraindata4$Stroke.Risk.Percentage - train_preds4)^2)

# Training R²
sst_train4 <- var(treetraindata4$Stroke.Risk.Percentage) * nrow(treetraindata4)
r2_train4 <- 1 - (mse_train4 / (sst_train4 / nrow(treetraindata4)))

cat("Training MSE:", mse_train4, "\n")
print(sqrt(mse_train4))
cat("Training R²:", r2_train4, "\n")

# ----------------------------
# 4. Evaluate on Test Data
# ----------------------------

# Predictions on test set
test_preds4 <- predict(final_tree_model4, treetestdata4)

# Test MSE
mse_test4 <- mean((treetestdata4$Stroke.Risk.Percentage - test_preds4)^2)

# Test R²
sst_test4 <- var(treetestdata4$Stroke.Risk.Percentage) * nrow(treetestdata4)
r2_test4 <- 1 - (mse_test4 / (sst_test4 / nrow(treetestdata4)))

cat("Test MSE:", mse_test4, "\n")
print(sqrt(mse_test4))
cat("Test R²:", r2_test4, "\n")

#Random Forest________________________________________________________________________________________
library(randomForest)
forest4 <- randomForest(Stroke.Risk.Percentage ~ ., data = treetraindata4, mtry = floor(sqrt(ncol(treetraindata4))),
                        ntree = 100, importance = TRUE)
importance(forest4)
varImpPlot(forest4)

# Predict on training set
predicted_train4 <- predict(forest4, treetraindata4)
actual_train4 <- treetraindata4$Stroke.Risk.Percentage

# Compute MSE and R² on training set
mse_train_forest4 <- mean((predicted_train4 - actual_train4)^2)
r_squared_train_forest4 <- 1 - sum((predicted_train4 - actual_train4)^2) / sum((actual_train4 - mean(actual_train4))^2)

cat("Training MSE (Random Forest):", mse_train_forest4, "\n")
print(sqrt(mse_train_forest4))
cat("Training R² (Random Forest):", r_squared_train_forest4, "\n")

predicted4 <- predict(forest4, treetestdata4)
actual4 <- treetestdata4$Stroke.Risk.Percentage
mse_forest4 <- mean((predicted4 - actual4)^2)
r_squared_forest4 <- 1 - sum((predicted4 - actual4)^2) / sum((actual4 - mean(actual4))^2)
cat("Tresting MSE (Random Forest):", mse_forest4, "\n")
print(sqrt(mse_forest4))
cat("Testing R² (Random Forest):", r_squared_forest4, "\n")


#cross validation on RF________________________________________________________________________________
mtry_grid4 <- expand.grid(mtry = seq(1, floor(sqrt(ncol(treetraindata4))), by = 1))
cv_rf4 <- train(Stroke.Risk.Percentage ~ ., 
                data = treetraindata4, 
                method = "rf", 
                trControl = train_control4,
                tuneGrid = mtry_grid4,
                ntree = 100,  
                importance = TRUE)

print(cv_rf4)  # Shows best mtry and RMSE
plot(cv_rf4)  # Visual of RMSE vs mtry

# Best mtry value
cv_rf4$bestTune

# Retrain final model with best mtry on full training data
final_rf4 <- randomForest(Stroke.Risk.Percentage ~ ., 
                          data = treetraindata4,
                          mtry = cv_rf4$bestTune$mtry,
                          ntree = 100, 
                          importance = TRUE)

importance(final_rf4)
varImpPlot(final_rf4)

# Training predictions
train_preds_final4 <- predict(final_rf4, treetraindata4)

# Training MSE
mse_train_final4 <- mean((train_preds_final4 - treetraindata4$Stroke.Risk.Percentage)^2)

# Training R²
sst_train4 <- var(treetraindata4$Stroke.Risk.Percentage) * nrow(treetraindata4)
r2_train_final4 <- 1 - (mse_train_final4 / (sst_train4 / nrow(treetraindata4)))

cat("Final MSE (Training):", mse_train_final4, "\n")
print(sqrt(mse_train_final4))
cat("Final R² (Training):", r2_train_final4, "\n")


# Test predictions
test_preds_final4 <- predict(final_rf4, treetestdata4)

# Test MSE
mse_test_final4 <- mean((test_preds_final4 - treetestdata4$Stroke.Risk.Percentage)^2)

# Test R²
sst_test4 <- var(treetestdata4$Stroke.Risk.Percentage) * nrow(treetestdata4)
r2_test_final4 <- 1 - (mse_test_final4 / (sst_test4 / nrow(treetestdata4)))

cat("Final MSE (Test):", mse_test_final4, "\n")
print(sqrt(mse_test_final4))
cat("Final R² (Test):", r2_test_final4, "\n")
