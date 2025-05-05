####
#### R code for Week #3: Linear Regression 
## 
## First, save the dataset "prostate.csv" in your laptop, say, 
##         in the local folder "C:/temp". 
##
## Data Set
fat <- read.table("fat.csv", header= TRUE, sep = ",")

##This dataset is from the textbook ESL, where the authors 
##  have split the data into the training and testing subsets
##  here we use their split to produce similar results 

n = dim(fat)[1];      ### total number of observations
n1 = round(n/10);     ### number of observations randomly selected for testing data
## To fix our ideas, let the following 25 rows of data as the testing subset:
flag = c(1,  21,  22,  57,  70,  88,  91,  94, 121, 127, 149, 151, 159, 162,
         164, 177, 179, 194, 206, 214, 215, 221, 240, 241, 243);
fat1train = fat[-flag,];    
fat1test  = fat[flag,];
## The true Y response for the testing subset
ytrue    <- fat1test$brozek; 

#Exploratory Data Analysis
library(DataExplorer)
length(fat1train$brozek)
get_mode <- function(v) {
  uniq_vals <- unique(v[!is.na(v)]) # Remove NAs and get unique values
  uniq_vals[which.max(tabulate(match(v, uniq_vals)))]
}
sapply(fat1train, get_mode)
mmm <- data.frame(
  Mean = sapply(fat1train, function(col) mean(col, na.rm = TRUE)),
  Median = sapply(fat1train, function(col) median(col, na.rm = TRUE)),
  Mode = sapply(fat1train, get_mode)
)
print(mmm)
plot_histogram(fat1train)
boxplot(fat1train, main = "Boxplots for Each Column")
plot_correlation(fat1train, maxcat = 5L)

### Below we consider seven (7) linear regression related models
### (1) Full model;  (2) The best subset model; (3) Stepwise variable selection with AIC
### (4) Ridge Regression; (5) LASSO; 
##  (6) Principal Component Regression, and (7) Parital Least Squares (PLS) Regression 
##
###   For each of these 7 models or methods, we fit to the training subset, 
###    and then compute its training and testing errors. 
##
##     Let us prepare to save all training and testing errors
MSEtrain <- NULL;
MSEtest  <- NULL; 

###
### (1) Linear regression with all predictors (Full Model)
###     This fits a full linear regression model on the training data
model1 <- lm( brozek ~ ., data = fat1train); 

## Model 1: Training error
MSEmod1train <-   mean( (resid(model1) )^2);
MSEtrain <- c(MSEtrain, MSEmod1train);
MSEmod1train
# Model 1: testing error 
pred1a <- predict(model1, fat1test[,2:18]);
MSEmod1test <-   (1/n1) * mean((pred1a - ytrue)^2);
MSEmod1test;
MSEtest <- c(MSEtest, MSEmod1test); 
#[1] 0.521274

### (2) Linear regression with the best subset model 
###  YOu need to first install the package "leaps"
library(leaps);
fat.leaps <- regsubsets(brozek ~ ., data= fat1train, nbest= 100, really.big= TRUE); 

## Record useful information from the output
fat.models <- summary(fat.leaps)$which;
fat.models.size <- as.numeric(attr(fat.models, "dimnames")[[1]]);
fat.models.rss <- summary(fat.leaps)$rss;

## 2A:  The following are to show the plots of all subset models 
##   and the best subset model for each subset size k 
plot(fat.models.size, fat.models.rss); 
## find the smallest RSS values for each subset size 
fat.models.best.rss <- tapply(fat.models.rss, fat.models.size, min); 
## Also add the results for the only intercept model
fat.model0 <- lm( brozek ~ 1, data = fat1train); 
fat.models.best.rss <- c( sum(resid(fat.model0)^2), fat.models.best.rss); 
## plot all RSS for all subset models and highlight the smallest values 
plot( 0:8, fat.models.best.rss, type = "b", col= "red", xlab="Subset Size k", ylab="Residual Sum-of-Square")
points(fat.models.size, fat.models.rss)

# 2B: What is the best subset with k=5
op2 <- which(fat.models.size == 5); 
flag2 <- op2[which.min(fat.models.rss[op2])]; 



## 2B We can auto-find the best subset with k=3
##   this way will be useful when doing cross-validation 
mod2selectedmodel <- fat.models[flag2,]; 
mod2Xname <- paste(names(mod2selectedmodel)[mod2selectedmodel][-1], collapse="+"); 
mod2form <- paste ("brozek ~", mod2Xname);
## To auto-fit the best subset model with k=3 to the data
model2 <- lm( as.formula(mod2form), data= fat1train); 
# Model 2: training error 
MSEmod2train <- mean(resid(model2)^2);
MSEmod2train
## save this training error to the overall training error vector 
MSEtrain <- c(MSEtrain, MSEmod2train);
MSEtrain;
## Model 2:  testing error 
pred2 <- predict(model2, fat1test[,2:18]);
MSEmod2test <-   (1/n1) * mean((pred2 - ytrue)^2);
MSEmod2test
MSEtest <- c(MSEtest, MSEmod2test);
MSEtest;
## Check the answer
##[1] 0.5212740 0.4005308

## As compared to the full model #1, the best subset model with K=3
##   has a larger training eror (0.521 vs 0.439),
##   but has a smaller testing error (0.400 vs 0.521). 


### (3) Linear regression with the stepwise variable selection 
###     that minimizes the AIC criterion 
##    This can done by using the "step()" function in R, 
##       but we need to build the full model first

model1 <- lm( brozek ~ ., data = fat1train); 
model3  <- step(model1); 

## If you want, you can see the coefficents of model3
round(coef(model3),3)
summary(model3)

## Model 3: training  and  testing errors 
MSEmod3train <- mean(resid(model3)^2);
MSEmod3train
pred3 <- predict(model3, fat1test[,2:18]);
MSEmod3test <-  (1/n1) * mean((pred3 - ytrue)^2);
MSEmod3test
MSEtrain <- c(MSEtrain, MSEmod3train);
MSEtrain; 
## [1] 0.4391998 0.5210112 0.4393627
MSEtest <- c(MSEtest, MSEmod3test);
## Check your answer 
MSEtest;
## [1] 0.5212740 0.4005308 0.5165135


### (4) Ridge regression (MASS: lm.ridge, mda: gen.ridge)
### We need to call the "MASS" library in R
### 
library(MASS);

## The following R code gives the ridge regression for all penality function lamdba
##  Note that you can change lambda value to other different range/stepwise 
fat.ridge <- lm.ridge( brozek ~ ., data = fat1train, lambda= seq(0,100,0.001));

## 4A. Ridge Regression plot how the \beta coefficients change with \lambda values 
##   Two equivalent ways to plot
plot(fat.ridge) 
### Or "matplot" to plot the columns of one matrix against the columns of another
matplot(fat.ridge$lambda, t(fat.ridge$coef), type="l", lty=1, 
        xlab=expression(lambda), ylab=expression(hat(beta)))

## 4B: We need to select the ridge regression model
##        with the optimal lambda value 
##     There are two ways to do so

## 4B(i) manually find the optimal lambda value
##    but this is infeasible for cross-validation 
select(fat.ridge)
## 
#modified HKB estimator is 3.355691 
#modified L-W estimator is 3.050708 
# smallest value of GCV  at 4.92 
#
# The output suggests that a good choice is lambda = 4.92, 
abline(v=0.003)
# Compare the ceofficients of ridge regression with lambda= 4.92
##  versus the full linear regression model #1 (i.e., with lambda = 0)
fat.ridge$coef[, which(fat.ridge$lambda == 0.003)]
fat.ridge$coef[, which(fat.ridge$lambda == 0)]

## 4B(ii) Auto-find the "index" for the optimal lambda value for Ridge regression 
##        and auto-compute the corresponding testing and testing error 
indexopt <-  which.min(fat.ridge$GCV);  

## If you want, the corresponding coefficients with respect to the optimal "index"
##  it is okay not to check it!
fat.ridge$coef[,indexopt]
## However, this coefficeints are for the the scaled/normalized data 
##      instead of original raw data 
## We need to transfer to the original data 
## Y = X \beta + \epsilon, and find the estimated \beta value 
##        for this "optimal" Ridge Regression Model
## For the estimated \beta, we need to sparate \beta_0 (intercept) with other \beta's
ridge.coeffs = fat.ridge$coef[,indexopt]/ fat.ridge$scales;
intercept = -sum( ridge.coeffs  * colMeans(fat1train[,2:18] )  )+ mean(fat1train[,1]);
## If you want to see the coefficients estimated from the Ridge Regression
##   on the original data scale
c(intercept, ridge.coeffs);

## Model 4 (Ridge): training errors 
yhat4train <- as.matrix( fat1train[,2:18]) %*% as.vector(ridge.coeffs) + intercept;
MSEmod4train <- mean((yhat4train - fat1train$brozek)^2); 
MSEmod4train
MSEtrain <- c(MSEtrain, MSEmod4train); 
MSEtrain
## [1]  0.4391998 0.5210112 0.4393627 0.4473617
## Model 4 (Ridge):  testing errors in the subset "test" 
pred4test <- as.matrix( fat1test[,2:18]) %*% as.vector(ridge.coeffs) + intercept;
MSEmod4test <-  (1/n1) * mean((pred4test - ytrue)^2); 
MSEmod4test
MSEtest <- c(MSEtest, MSEmod4test);
MSEtest;
## [1] 0.5212740 0.4005308 0.5165135 0.4943531


## Model (5): LASSO 
## IMPORTANT: You need to install the R package "lars" beforehand
##
library(lars)
fat.lars <- lars( as.matrix(fat1train[,2:18]), fat1train[,1], type= "lasso", trace= TRUE);

## 5A: some useful plots for LASSO for all penalty parameters \lambda 
plot(fat.lars)

## 5B: choose the optimal \lambda value that minimizes Mellon's Cp criterion 
Cp1  <- summary(fat.lars)$Cp;
index1 <- which.min(Cp1);

## 5B(i) if you want to see the beta coefficient values (except the intercepts)
##   There are three equivalent ways
##    the first two are directly from the lars algorithm
coef(fat.lars)[index1,]
fat.lars$beta[index1,]
##   the third way is to get the coefficients via prediction function 
lasso.lambda <- fat.lars$lambda[index1]
coef.lars1 <- predict(fat.lars, s=lasso.lambda, type="coef", mode="lambda")
coef.lars1$coef
## Can you get the intercept value? 
##  \beta0 = mean(Y) - mean(X)*\beta of training data
##       for all linear models including LASSO
LASSOintercept = mean(fat1train[,1]) -sum( coef.lars1$coef  * colMeans(fat1train[,2:18] ));
c(LASSOintercept, coef.lars1$coef)

## Model 5:  training error for lasso
## 
pred5train  <- predict(fat.lars, as.matrix(fat1train[,2:18]), s=lasso.lambda, type="fit", mode="lambda");
yhat5train <- pred5train$fit; 
MSEmod5train <- mean((yhat5train - fat1train$brozek)^2); 
MSEmod5train
MSEtrain <- c(MSEtrain, MSEmod5train); 
MSEtrain
# [1] 0.4391998 0.5210112 0.4393627 0.4473617 0.4398267
##
## Model 5:  training error for lasso  
pred5test <- predict(fat.lars, as.matrix(fat1test[,2:18]), s=lasso.lambda, type="fit", mode="lambda");
yhat5test <- pred5test$fit; 
MSEmod5test <- (1/n1) * mean( (yhat5test - fat1test$brozek)^2); 
MSEmod5test
MSEtest <- c(MSEtest, MSEmod5test); 
MSEtest;
## Check your answer:
## [1] 0.5212740 0.4005308 0.5165135 0.4943531 0.5074249


#### Model 6: Principal Component Regression (PCR) 
##
## We can either manually conduct PCR by ourselves 
##   or use R package such as "pls" to auto-run PCR for us
##
## For purpose of learning, let us first conduct the manual run of PCR
##  6A: Manual PCR: 
##  6A (i) some fun plots for PCA of training data
trainpca <- prcomp(fat1train[,2:18]);  
##
## 6A(ii)  Examine the square root of eigenvalues
## Most variation in the predictors can be explained 
## in the first a few dimensions
trainpca$sdev
round(trainpca$sdev,2)
### 6A (iii) Eigenvectors are in oj$rotation
### the dim of vectors is 8
###
matplot(2:18, trainpca$rot[,1:3], type ="l", xlab="", ylab="")
matplot(2:18, trainpca$rot[,1:5], type ="l", xlab="", ylab="")
##
## 6A (iv) Choose a number beyond which all e. values are relatively small 
plot(trainpca$sdev,type="l", ylab="SD of PC", xlab="PC number")
##
## 6A (v) An an example, suppose we want to do Regression on the first 4 PCs
## Get Pcs from obj$x
modelpca <- lm(brozek ~ trainpca$x[,1:4], data = fat1train)
##
## 6A (vi) note that this is on the PC space (denote by Z), with model Y= Z\gamma + epsilon
## Since the PCs Z= X U for the original data, this yields to 
## Y= X (U\gamma) + epsilon,
## which is the form Y=X\beta + epsilon in the original data space 
##  with \beta = U \gamma. 
beta.pca <- trainpca$rot[,1:4] %*% modelpca$coef[-1]; 
##
## 6A (vii) as a comparion of \beta for PCA, OLS, Ridge and LASSO
##   without intercepts, all on the original data scale
cbind(beta.pca, coef(model1)[-1], ridge.coeffs, coef.lars1$coef)
##
### 6A(viii) Prediciton for PCA
### To do so, we need to first standardize the training or testing data, 
### For any new data X, we need to impose the center as in the training data
###  This requires us to subtract the column mean of training from the test data
xmean <- apply(fat1train[,2:18], 2, mean); 
xtesttransform <- as.matrix(sweep(fat1test[,2:18], 2, xmean)); 
##
## 6A (iX) New testing data X on the four PCs
xtestPC <-  xtesttransform %*% trainpca$rot[,1:4]; 
##
## 6A (X) the Predicted Y
ypred6 <- cbind(1, xtestPC) %*% modelpca$coef;  
## 
## In practice, one must choose the number of PC carefully.
##   Use validation dataset to choose it. Or Use cross-Validation 
##  This can be done use the R package, say "pls"
##  in the "pls", use the K-fold CV -- default; divide the data into K=10 parts 
##
## 6B: auto-run PCR
##
## You need to first install the R package "pls" below
##
library(pls)
## 6B(i): call the pcr function to run the linear regression on all possible # of PCs.
##
fat.pca <- pcr(brozek~., data=fat1train, validation="CV");  
## 
## 6B(ii) You can have some plots to see the effects on the number of PCs 
validationplot(fat.pca);
summary(fat.pca); 
## The minimum occurs at 8 components
## so for this dataset, maybe we should use full data
##
### 6B(iii) How to auto-select # of components
##     automatically optimazation by PCR based on the cross-validation
##     It chooses the optimal # of components 
ncompopt <- which.min(fat.pca$validation$adj);
ncompopt
## 
## 6B(iv) Training Error with the optimal choice of PCs
ypred6train <- predict(fat.pca, ncomp = ncompopt, newdata = fat1train[2:18]); 
MSEmod6train <- mean( (ypred6train - fat1train$brozek)^2); 
MSEmod6train
MSEtrain <- c(MSEtrain, MSEmod6train); 
MSEtrain;
## 6B(v) Testing Error with the optimal choice of PCs
ypred6test <- predict(fat.pca, ncomp = ncompopt, newdata = fat1test[2:18]); 
MSEmod6test <- (1/n1) * mean( (ypred6test - fat1test$brozek)^2); 
MSEmod6test
MSEtest <- c(MSEtest, MSEmod6test); 
MSEtest;
## Check your answer:
## [1] 0.5212740 0.4005308 0.5165135 0.4943531 0.5074249 0.5212740
##
## Fo this specific example, the optimal # of PC
##         ncompopt = 8, which is the full dimension of the original data
##   and thus the PCR reduces to the full model!!!


### Model 7. Partial Least Squares (PLS) Regression 
###
###  The idea is the same as the PCR and can be done by "pls" package
###  You need to call the fuction "plsr"  if you the code standalone 
#  library(pls)
fat.pls <- plsr(brozek ~ ., data = fat1train, validation="CV");

### 7(i) auto-select the optimal # of components of PLS 
## choose the optimal # of components  
mod7ncompopt <- which.min(fat.pls$validation$adj);
## The opt # of components, it turns out to be 8 for this dataset,
##       and thus PLS also reduces to the full model!!!    
 
# 7(ii) Training Error with the optimal choice of "mod7ncompopt" 
# note that the prediction is from "prostate.pls" with "mod7ncompopt" 
ypred7train <- predict(fat.pls, ncomp = mod7ncompopt, newdata = fat1train[2:18]); 
MSEmod7train <- mean( (ypred7train - fat1train$brozek)^2); 
MSEmod7train
MSEtrain <- c(MSEtrain, MSEmod7train); 
MSEtrain;
## 7(iii) Testing Error with the optimal choice of "mod7ncompopt" 
ypred7test <- predict(fat.pls, ncomp = mod7ncompopt, newdata = fat1test[2:18]); 
MSEmod7test <- (1/n1) * mean( (ypred7test - fat1test$brozek)^2); 
MSEmod7test
MSEtest <- c(MSEtest, MSEmod7test); 
MSEtest;

## Check your answers
MSEtrain 
## Training errors of these 7 models/methods
#[1] 0.4391998 0.5210112 0.4393627 0.4473617 0.4398267 0.4391998 0.4391998
MSEtest
## Testing errors of these 7 models/methods
#[1] 0.5212740 0.4005308 0.5165135 0.4943531 0.5074249 0.5212740 0.5212740
##
## For this specific dataset, PCR and PLS reduce to the full model!!!

### Part (e): the following R code might be useful, and feel free to modify it.
###    save the TE values for all models in all $B=100$ loops
B= 100;
TEALL = NULL;
set.seed(7406);
### number of loops
### Final TE values
### You might want to set the seed for randomization
for (b in 1:B){
  ### randomly select 25 observations as testing data in each loop
  flag <- sort(sample(1:n, n1));
  fattrain <- fat[-flag,];
  fattest  <- fat[flag,];
  truey <- fattest$brozek;

  ### you can write your own R code here to first fit each model to "fattrain"
  ### then get the testing error (TE) values on the testing data "fattest"
  ### Suppose that you save the TE values for these five models as
  ###   te1, te2, te3, te4, te5, te6, te7, respectively, within this loop
  ###   Then you can save these 5 Testing Error values by using the R code
  ###
  # Model 1: Linear regression with all predictors
  mod1 <- lm( brozek ~ ., data = fattrain); 
  pred1 <- predict(mod1, fattest[,2:18]);
  te1 <-   (1/n1) * mean((pred1 - truey)^2);
  
  # Model 2: Linear regression with the best subset of k = 5 predictors
  library(leaps);
  fat.l <- regsubsets(brozek ~ ., data= fattrain, nbest= 100, really.big= TRUE); 
  fat.m <- summary(fat.l)$which
  fat.m.size <- as.numeric(attr(fat.m, "dimnames")[[1]]);
  fat.m.rss <- summary(fat.l)$rss;
  op <- which(fat.m.size == 5); 
  flagk <- op2[which.min(fat.m.rss[op])]; 
  mod2selectedmod <- fat.models[flagk,]; 
  mod2Xn <- paste(names(mod2selectedmod)[mod2selectedmod][-1], collapse="+"); 
  mod2f <- paste ("brozek ~", mod2Xn);
  mod2 <- lm( as.formula(mod2f), data= fattrain); 
  pred2b <- predict(mod2, fattest[,2:18]);
  te2 <-   (1/n1) * mean((pred2b - truey)^2);
  
  # Model 3: Linear regression with stepwise selection using AIC
  mod3  <- step(mod1); 
  pred3b <- predict(mod3, fattest[,2:18]);
  te3 <-  (1/n1) * mean((pred3b - truey)^2);
  
  # Model 4: Ridge regression
  
  library(MASS);
  fat.rid <- lm.ridge( brozek ~ ., data = fattrain, lambda= seq(0,100,0.001));
  idxopt <-  which.min(fat.rid$GCV);
  fat.rid$coef[,idxopt]
  rid.coeffs = fat.rid$coef[,idxopt]/ fat.rid$scales;
  inter = -sum( rid.coeffs  * colMeans(fattrain[,2:18] )  )+ mean(fattrain[,1]);
  c(inter, rid.coeffs);
  pred4testb <- as.matrix( fattest[,2:18]) %*% as.vector(rid.coeffs) + inter;
  te4 <-  (1/n1) * mean((pred4testb - truey)^2); 
 
  
  # Model 5: Lasso regression
  
  library(glmnet)
  X_train <- as.matrix(fattrain[, 2:18]) # Predictor matrix for training
  y_train <- fattrain[, 1]              # Response variable for training
  X_test <- as.matrix(fattest[, 2:18])  # Predictor matrix for testing
  y_test <- truey                       # True response values for the test set
  cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)  # alpha = 1 for Lasso
  optimal_lambda <- cv_lasso$lambda.min
  pred5testb <- predict(cv_lasso, newx = X_test, s = optimal_lambda)
  te5 <- (1/n1) * mean((pred5testb - y_test)^2)

  # Model 6: Principal Component Regression (PCR)
  
  library(pls)
  fat.pc <- pcr(brozek~., data=fattrain, validation="CV");  
  validationplot(fat.pc);
  summary(fat.pc); 
  ncompop <- which.min(fat.pc$validation$adj);
  ypred6testb <- predict(fat.pc, ncomp = ncompop, newdata = fattest[2:18]); 
  te6 <- (1/n1) * mean( (ypred6testb - truey)^2); 
  
  # Model 7: Partial Least Squares Regression (PLSR)
  fat.pl <- plsr(brozek ~ ., data = fattrain, validation="CV");
  mod7ncompop <- which.min(fat.pl$validation$adj);

  ## 7(iii) Testing Error with the optimal choice of "mod7ncompopt" 
  ypred7testb <- predict(fat.pl, ncomp = mod7ncompop, newdata = fattest[2:18]); 
  te7 <- (1/n1) * mean( (ypred7testb - truey)^2); 

  
  TEALL = rbind( TEALL, cbind(te1, te2, te3, te4, te5, te6, te7) );
}
dim(TEALL);  ### This should be a Bx7 matrices
### if you want, you can change the column name of TEALL
colnames(TEALL) <- c("mod1", "mod2", "mod3", "mod4", "mod5", "mod6", "mod7");
## You can report the sample mean and sample variances for the seven models
apply(TEALL, 2, mean);
apply(TEALL, 2, var);
### END ###

