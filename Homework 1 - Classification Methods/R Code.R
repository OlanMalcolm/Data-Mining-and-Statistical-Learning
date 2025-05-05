## 1. Read Training data
ziptrain <- read.table(file= "zip.train.csv", sep = ",");
ziptrain27  <- subset(ziptrain, ziptrain[,1]==2 | ziptrain[,1]==7);
## some sample Exploratory Data Analysis
dim(ziptrain27);    ## 1376 257
sum(ziptrain27[,1] == 2);
sum(ziptrain27[,1] == 7);
barplot(table(ziptrain27$V1), main="Distribution of Digits in Data Set", col = 'purple')
summary(ziptrain27);
cor_matrix <- round(cor(ziptrain27[, -1]),2);
cor_matrix
install.packages("ggcorrplot")
library(ggcorrplot)
# Visualize with ggcorrplot
ggcorrplot(cor_matrix)
## To see the letter picture of the 5-th row by changing the row observation to a matrix
rowindex = 5;  ## You can try other "rowindex" values to see other rows
ziptrain27[rowindex,1];
Xval = t(matrix(data.matrix(ziptrain27[,-1])[rowindex,],byrow=TRUE,16,16)[16:1,]);
image(Xval,col=gray(0:32/32),axes=FALSE) ## Also try "col=gray(0:32/32)"
### 2. Build Classification Rules
### linear Regression
mod1 <- lm( V1 ~ . , data= ziptrain27);
pred1.train <- predict.lm(mod1, ziptrain27[,-1]);
pred1.train
y1pred.train <- 2 + 5*(pred1.train >= 4.5);
## Note that we predict Y1 to $2$ and $7$,
##   depending on the indicator variable whether pred1.train >= 4.5 = (2+7)/2.
train_error_lm <- mean( y1pred.train  != ziptrain27[,1]);
train_error_lm
## KNN
library(class)
# Initialize a vector to store training errors for each k
k_values <- c(1, 3, 5, 7, 9, 11, 13, 15)
train_errors_knn <- numeric(length(k_values))

# Compute training errors for each k
for (i in seq_along(k_values)) {
  k <- k_values[i]
  ypred_knn <- knn(ziptrain27[, -1], ziptrain27[, -1], ziptrain27[, 1], k = k)
  train_errors_knn[i] <- mean(ypred_knn != ziptrain27[, 1])
}

# Print training errors for each k
train_results <- data.frame(k = k_values, Training_Error = train_errors_knn)
print(train_results)

### 3. Testing Error
### read testing data
ziptest <- read.table(file="zip.test.csv", sep = ",");
ziptest27  <- subset(ziptest, ziptest[,1]==2 | ziptest[,1]==7);
dim(ziptest27) ##345 257
## Testing error of KNN, and you can change the k values.
xnew2 <- ziptest27[,-1];         ## xnew2 is the X variables of the  "testing" data
ytest <- ziptest27[, 1]   # Response variable

# Predict on the testing data using the linear regression model
pred1.test <- predict(mod1, ziptest27)

# Convert continuous predictions to discrete classes (2 or 7)
y1pred.test <- 2 + 5 * (pred1.test >= 4.5)

# Calculate testing error
test_error_lm <- mean(y1pred.test != ytest)
test_error_lm

# Initialize a vector to store testing errors for each k
test_errors_knn <- numeric(length(k_values))

# Compute testing errors for each k
for (i in seq_along(k_values)) {
  k <- k_values[i]
  ypred_knn_test <- knn(ziptrain27[, -1], xnew2, ziptrain27[, 1], k = k)
  test_errors_knn[i] <- mean(ypred_knn_test != ytest)
}

# Print testing errors for each k
test_results <- data.frame(k = k_values, Testing_Error = test_errors_knn)
test_results

### 4. Cross-Validation
### The following R code might be useful, but you need to modify it.
zip27full =  rbind(ziptrain27,  ziptest27)      ### combine to a full data set
n1 = 1376;   # training set sample size
n2= 345;     # testing set sample size
n = dim(zip27full)[1];    ## the total sample size
set.seed(7406);   ### set the seed for randomization
###    Initialize the TE values for all models in all $B=100$ loops
B= 100;            ### number of loops
TEALL = NULL;      ### Final TE values
for (b in 1:B){
  ### randomly select n1 observations as a new training  subset in each loop
  flag <- sort(sample(1:n, n1));
  zip27traintemp <- zip27full[flag,];  ## temp training set for CV
  zip27testtemp  <- zip27full[-flag,]; ## temp testing set for CV
  ### you need to write your own R code here to first fit each model to "zip27traintemp"
  ### then get the testing error (TE) values on the testing data "zip27testtemp"

  mod <- lm(V1 ~ ., data = zip27traintemp)  # Fit linear regression
  pred <- predict(mod, zip27testtemp[,-1]) # Predict on the testing set
  te0 <- mean((2 + 5 * (pred >= 4.5)) != zip27testtemp[, 1]) # Testing error

  te1 <- mean(knn(zip27traintemp[,-1], zip27testtemp[,-1], zip27traintemp[, 1], k = 1) != zip27testtemp[, 1])
  te2 <- mean(knn(zip27traintemp[,-1], zip27testtemp[,-1], zip27traintemp[, 1], k = 3) != zip27testtemp[, 1])
  te3 <- mean(knn(zip27traintemp[,-1], zip27testtemp[,-1], zip27traintemp[, 1], k = 5) != zip27testtemp[, 1])
  te4 <- mean(knn(zip27traintemp[,-1], zip27testtemp[,-1], zip27traintemp[, 1], k = 7) != zip27testtemp[, 1])
  te5 <- mean(knn(zip27traintemp[,-1], zip27testtemp[,-1], zip27traintemp[, 1], k = 9) != zip27testtemp[, 1])
  te6 <- mean(knn(zip27traintemp[,-1], zip27testtemp[,-1], zip27traintemp[, 1], k = 11) != zip27testtemp[, 1])
  te7 <- mean(knn(zip27traintemp[,-1], zip27testtemp[,-1], zip27traintemp[, 1], k = 13) != zip27testtemp[, 1])
  te8 <- mean(knn(zip27traintemp[,-1], zip27testtemp[,-1], zip27traintemp[, 1], k = 15) != zip27testtemp[, 1])
  
  ### IMPORTANT: when copying your codes in (2) and (3), please change to
  ###      these temp datasets, "zip27traintemp" and "zip27testtemp" !!!
  ###
  ### Suppose you save the TE values for these 9 methods (1 linear regression and 8 KNN) as
  ###  te0, te1, te2, te3, te4, te5, te6, te7, te8 respectively, within this loop
  ###   Then you can save these $9$ Testing Error values by using the R code
  ###   Note that the code is not necessary the most efficient
  TEALL = rbind( TEALL, cbind(te0, te1, te2, te3, te4, te5, te6, te7, te8) );
}
TEALL
### Of course, you can also get the training errors if you want
dim(TEALL);  ### This should be a Bx9 matrices
### if you want, you can change the column name of TEALL
colnames(TEALL) <- c("linearRegression", "KNN1", "KNN3", "KNN5", "KNN7",
                     "KNN9", "KNN11", "KNN13", "KNN15");
## You can report the sample mean/variances of the testing errors so as to compare these models
apply(TEALL, 2, mean);
apply(TEALL, 2, var);

### END ###
