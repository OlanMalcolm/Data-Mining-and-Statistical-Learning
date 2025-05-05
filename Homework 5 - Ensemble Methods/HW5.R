


### (A). Data Preparation 
## Read Data 
spam <- read.table(file= "diabetes_dataset.csv", sep = ",", header = TRUE)

dim(spam)   # 9538 17


#EDA
library(DataExplorer)
plot_histogram(spam)
plot_correlation(spam)

## Split to training and testing subset 
set.seed(123)
spam$FamilyHistory <- NULL
flag <- sort(sample(9538,2862, replace = FALSE))
spamtrain <- spam[-flag,]
spamtest <- spam[flag,]
## Extra the true response value for training and testing data
y1    <- spamtrain$Outcome;
y2    <- spamtest$Outcome;


## (B) Boosting 
## You need to first install this R package before using it
library(gbm)

# 
gbm.spam1 <- gbm(Outcome ~ .,data=spamtrain,
                 distribution = 'bernoulli',
                 n.trees = 5000, 
                 shrinkage = 0.01, 
                 interaction.depth = 4,
                 cv.folds = 10)
                  
## Model Inspection 
## Find the estimated optimal number of iterations
perf_gbm1 = gbm.perf(gbm.spam1, method="cv") 
perf_gbm1

pred1gbm <- predict(gbm.spam1, newdata = spamtrain, n.trees = perf_gbm1, type = "response")

## summary model
## Which variances are important
summary(gbm.spam1)


## Make Prediction
## use "predict" to find the training or testing error

## Training error
y1hat <- ifelse(pred1gbm < 0.5, 0, 1)
sum(y1hat != y1)/length(y1)  

## Testing Error
y2hat <- ifelse(predict(gbm.spam1, newdata = spamtest, n.trees=perf_gbm1, type="response") < 0.5, 0, 1)
mean(y2hat != y2) 


## A comparison with other methods
## Testing errors of several algorithms on the spam dataset:
#A. Logistic regression: 0.1022135 
modA <- step(glm(Outcome ~ ., data = spamtrain));

summary(modA)
# Training Error for Logistic Regression
y1hatA <- ifelse(predict(modA, spamtrain[,-16], type="response") < 0.5, 0, 1)
sum(y1hatA != y1) / length(y1)


y2hatA <- ifelse(predict(modA, spamtest[,-16], type="response" ) < 0.5, 0, 1)
sum(y2hatA != y2)/length(y2) 

#B.Linear Discriminant Analysis : 0.1041667 
library(MASS)
modB <- lda(spamtrain[,1:15], spamtrain[,16])
summary(modB)
print(modB)
# Training Error for LDA
y1hatB <- predict(modB, spamtrain[, -16])$class
mean(y1hatB != y1)


y2hatB <- predict(modB, spamtest[,-16])$class
mean( y2hatB  != y2)

## C. Naive Bayes (with full X). Testing error = 0.313151
library(e1071)
modC <- naiveBayes(as.factor(Outcome) ~. , data = spamtrain)
print(modC)

# Training Error for Naive Bayes
y1hatC <- predict(modC, newdata = spamtrain)
mean(y1hatC != y1)

y2hatC <- predict(modC, newdata = spamtest)
mean( y2hatC != y2) 

## D. KNN
library(class)
# Extract features and target variable from training data
train_features <- spamtrain[, -which(names(spamtrain) == "Outcome")]
train_target <- spamtrain$Outcome

# Extract features from test data
# Separate features and target in the test data
test_features <- spamtest[, -which(names(spamtest) == "Outcome")]  # Remove target from test data
test_target <- spamtest$Outcome  # Target variable in the test data

# Initialize a vector to store error rates for different k values
k_values <- 1:20

error_rates_train <- numeric(length(k_values))
error_rates <- numeric(length(k_values))

for (k in k_values) {
  # Apply KNN model with current k
  predictions_train <- knn(train = train_features, test = train_features, cl = train_target, k = k)
  
  # Calculate the training error rate for the current k
  error_rate_train <- mean(predictions_train != train_target)
  
  # Store the training error rate for the current k
  error_rates_train[k] <- error_rate_train
}

# Display the training error rates for each k value
data.frame(k = k_values, training_error_rate = error_rates_train)


# Loop over each k value and calculate the error rate
for (k in k_values) {
  # Apply KNN model with current k
  predictions <- knn(train = train_features, test = test_features, cl = train_target, k = k)
  
  # Calculate the error rate for the current k
  error_rate <- mean(predictions != test_target)
  
  # Store the error rate for the current k
  error_rates[k] <- error_rate
}

# Display the error rates for each k value
data.frame(k = k_values, error_rate = error_rates)

#E: a single Tree: 0.1015625
library(rpart)
library(rpart.plot)

modE0 <- rpart(Outcome ~ .,data=spamtrain, method="class", 
                     parms=list(split="gini"))
opt <- which.min(modE0$cptable[, "xerror"]); 
cp1 <- modE0$cptable[opt, "CP"];
modE <- prune(modE0,cp=cp1);
rpart.plot(modE, type=2, extra=101, fallen.leaves=TRUE, cex=0.8, main="Pruned Decision Tree for Diabetes")
# Training Error for Decision Tree
y1hatE <- predict(modE, spamtrain[,-16], type="class")
mean(y1hatE != y1)

y2hatE <-  predict(modE, spamtest[,-16],type="class")
mean(y2hatE != y2)

#F: Random Forest: 0.04166667
library(randomForest)
modF <- randomForest(as.factor(Outcome) ~., data=spamtrain, 
                    importance=TRUE)

y1hatF <- predict(modF, spamtrain, type='class')
mean(y1hatF != y1)

y2hatF = predict(modF, spamtest, type='class')
mean(y2hatF != y2)



library(caret)
# (B) Set up cross-validation control
train_control <- trainControl(method = "cv", number = 10)

# (C) Models with Cross-validation

# 1. Boosting
spamtrain$Outcome <- as.factor(spamtrain$Outcome)
spamtest$Outcome <- as.factor(spamtest$Outcome)

tune_grid <- expand.grid(
  n.trees = 351,                      # Number of trees to use (same as original)
  shrinkage = 0.01,                    # Learning rate (same as original)
  interaction.depth = 4,   # Tree depth (same as original)
  n.minobsinnode = 10
)

# Boosting model with cross-validation
gbm_model <- train(Outcome ~ ., data = spamtrain, 
                   method = "gbm", 
                   trControl = train_control, 
                   verbose = FALSE,
                   distribution = "bernoulli",   # Bernoulli for binary classification
                   tuneGrid = tune_grid)         # Use the defined tuning grid

# Optionally, print a summary of the model (best performance, optimal parameters)
summary(gbm_model)

# 2. Logistic Regression
logit_model <- train(Outcome ~ ., data = spamtrain, 
                     method = "glm", 
                     trControl = train_control)
summary(logit_model)

# 3. Linear Discriminant Analysis (LDA)
lda_model <- train(Outcome ~ ., data = spamtrain, 
                   method = "lda", 
                   trControl = train_control)
summary(lda_model)

# 4. Naive Bayes
nb_model <- train(Outcome ~ ., data = spamtrain, 
                  method = "naive_bayes", 
                  trControl = train_control)
summary(nb_model)

# 5. KNN
# Define the range of k values to test
k_values <- 1:20

# Initialize a vector to store error rates for each k
cv_error_rates <- numeric(length(k_values))

# Loop over each k value to perform cross-validation
for (k in k_values) {
  # Train the KNN model for each k with 10-fold cross-validation
  knn_model <- train(Outcome ~ ., 
                     data = spamtrain, 
                     method = "knn", 
                     tuneGrid = data.frame(k = k), 
                     trControl = train_control)
  
  # Extract the cross-validated error rate for the current k
  cv_error_rates[k] <- min(knn_model$results$Accuracy) # Accuracy is used here, or you can use ErrorRate = 1 - Accuracy
  
}

# Display the cross-validation errors for each k
cv_error_rates_df <- data.frame(k = k_values, cv_error_rate = 1 - cv_error_rates)
print(cv_error_rates_df)

# 6. Decision Tree (rpart)

# Train the decision tree model


# (2) Train the Model with Cross-Validation
# We're specifying the method as "rpart" (decision tree) and the formula `Outcome ~ .`
# We'll use the trainControl for cross-validation and search over a range of cp values.
tune_grid <- expand.grid(cp = seq(0.001, 0.1, by = 0.001))  # Search for a good CP value

tree_model_cv <- train(Outcome ~ ., data = spamtrain, 
                       method = "rpart", 
                       trControl = train_control,   # Cross-validation control
                       tuneGrid = tune_grid,        # CP tuning grid
                       parms = list(split = "gini"))  # Split criterion: "gini"

# (3) Inspect the Cross-Validation Results
# Best complexity parameter (cp) and model performance
print(tree_model_cv)

# (4) Plot the model's performance
# Visualize the effect of different complexity parameters on the performance
plot(tree_model_cv)

# (5) Use the best model from cross-validation for predictions
best_cp <- tree_model_cv$bestTune$cp  # Get the best CP value from cross-validation
best_tree_model <- tree_model_cv$finalModel  # The final decision tree model based on best CP

rpart.plot(best_tree_model, type = 2, extra = 101, fallen.leaves = TRUE, cex = 0.8, main = "CV Pruned Decision Tree for Diabetes")

# 7. Random Forest
rf_model <- train(Outcome ~ ., data = spamtrain, 
                  method = "rf", 
                  trControl = train_control)




# (E) Final Model Evaluation on Test Set
# Use the best-performing model (based on cross-validation results) to make predictions on the test set.

rf_predictions <- predict(rf_model, newdata = spamtest)
rf_testing_error <- mean(rf_predictions != y2)
print(paste("Random Forest Testing Error:", rf_testing_error))

# Similarly, for other models:
# For Boosting (gbm)
gbm_predictions <- predict(gbm_model, newdata = spamtest) 
gbm_testing_error <- mean(gbm_predictions != y2)
print(paste("Boosting Testing Error:", gbm_testing_error))

# For Logistic Regression (logit_model)
logit_predictions <- predict(logit_model, newdata = spamtest[,-16]) 
logit_testing_error <- mean(logit_predictions != y2)
print(paste("Logistic Regression Testing Error:", logit_testing_error))

# For Linear Discriminant Analysis (lda_model)
lda_predictions <- predict(lda_model, newdata = spamtest[,-16]) 
lda_testing_error <- mean(lda_predictions != y2)
print(paste("LDA Testing Error:", lda_testing_error))

# For Naive Bayes (nb_model)
nb_predictions <- predict(nb_model, newdata = spamtest[,-16]) 
nb_testing_error <- mean(nb_predictions != y2)
print(paste("Naive Bayes Testing Error:", nb_testing_error))

# For KNN (knn_model)
knn_predictions <- predict(knn_model, newdata = spamtest[,-16])
knn_testing_error <- mean(knn_predictions != y2)
print(paste("KNN Testing Error:", knn_testing_error))


y2hatE <- predict(best_tree_model, spamtest[,-16], type = "class")
testing_error <- mean(y2hatE != y2)  # Calculate testing error
print(paste("Testing Error Rate:", testing_error))

