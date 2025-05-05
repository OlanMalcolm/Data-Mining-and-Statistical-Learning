#!/usr/bin/env python
# coding: utf-8

# Load Data

# In[469]:


import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt  
import numpy as np
from sklearn.model_selection import train_test_split 

auto1 = pd.read_csv("Auto.csv")
auto1.head()


# In[470]:


auto = auto1.copy()
auto['mpg'] = (auto['mpg'] > auto['mpg'].median()).astype('int')
display(auto)


# EDA

# In[472]:


mean_values = auto1.mean()       # Mean of each column
median_values = auto1.median()   # Median of each column
mode_values = auto1.mode().iloc[0]

print("Mean:\n", mean_values)
print("Median:\n", median_values)
print("Mode:\n", mode_values)


# In[473]:


auto1.describe()


# In[474]:


counts = auto.groupby('mpg').size()
print(counts)
auto['mpg'].value_counts().sort_index().plot(kind='bar', figsize=(10, 5))
plt.xlabel('MPG')
plt.ylabel('Count')
plt.title('Distribution of MPG')
plt.xticks(rotation=0)
plt.show()


# In[475]:


sns.set_style("darkgrid")

numerical_columns = auto.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(auto[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(auto[feature].skew(), 2)}")

plt.tight_layout()
plt.show()


# In[476]:


auto.boxplot(figsize=(10, 6))  # Creates boxplots for all numerical columns
plt.title("Boxplots for All Columns")
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()


# In[477]:


auto.boxplot(column = 'horsepower')
plt.title("Boxplot for Horsepower")
plt.show()
auto.boxplot(column = 'acceleration')
plt.title("Boxplot for Acceleration")
plt.show()


# In[478]:


auto.corr() 
sns.heatmap(auto.corr(), annot=True, cmap="coolwarm")  # Heatmap of correlations
plt.show()


# Models

# In[480]:


X = auto.drop(columns=['mpg'])  # Features
y = auto['mpg']  # Target variable

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# In[481]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

#LDA

# LDA Model
lda = LinearDiscriminantAnalysis()

# Fit on the training data
lda.fit(X_train, y_train)

# Predict on the training data
y_train_pred = lda.predict(X_train)

# Calculate training MSE
training_mse = mean_squared_error(y_train, y_train_pred)
print(f"Training Mean Squared Error: {training_mse:.6f}")

# Predict on the test data
y_test_pred = lda.predict(X_test)

# Calculate testing MSE
testing_mse = mean_squared_error(y_test, y_test_pred)
print(f"Testing Mean Squared Error: {testing_mse:.6f}")

# Perform 10-fold cross-validation on the training data
cv_scores = cross_val_score(lda, X, y, cv=10, scoring='neg_mean_squared_error')

# Negate the scores to get the positive MSE values
cv_mse_scores = -cv_scores

# Compute the mean of MSE across all folds
average_mse = np.mean(cv_mse_scores)
# Compute the variance of MSE across all folds
variance_mse = np.var(cv_mse_scores)

# Print the results
print(f"Average Mean Squared Error from Cross-Validation: {average_mse:.6f}")
print(f"Variance of Mean Squared Error from Cross-Validation: {variance_mse:.6f}")


# In[482]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#QDA
qda = QuadraticDiscriminantAnalysis()
# Fit on the training data
qda.fit(X_train, y_train)

# Predict on the training data
y_train_pred = qda.predict(X_train)

# Calculate training MSE
training_mse = mean_squared_error(y_train, y_train_pred)
print(f"Training Mean Squared Error: {training_mse:.6f}")

# Predict on the test data
y_test_pred = qda.predict(X_test)

# Calculate testing MSE
testing_mse = mean_squared_error(y_test, y_test_pred)
print(f"Testing Mean Squared Error: {testing_mse:.6f}")

# Perform 10-fold cross-validation on the training data
cv_scores = cross_val_score(qda, X, y, cv=10, scoring='neg_mean_squared_error')

# Negate the scores to get the positive MSE values
cv_mse_scores = -cv_scores

# Compute the mean of MSE across all folds
average_mse = np.mean(cv_mse_scores)
# Compute the variance of MSE across all folds
variance_mse = np.var(cv_mse_scores)

# Print the results
print(f"Average Mean Squared Error from Cross-Validation: {average_mse:.6f}")
print(f"Variance of Mean Squared Error from Cross-Validation: {variance_mse:.6f}")


# In[483]:


from sklearn.naive_bayes import GaussianNB

#Naive Bayes
gnb = GaussianNB()
# Fit on the training data
gnb.fit(X_train, y_train)

# Predict on the training data
y_train_pred = gnb.predict(X_train)

# Calculate training MSE
training_mse = mean_squared_error(y_train, y_train_pred)
print(f"Training Mean Squared Error: {training_mse:.6f}")

# Predict on the test data
y_test_pred = gnb.predict(X_test)

# Calculate testing MSE
testing_mse = mean_squared_error(y_test, y_test_pred)
print(f"Testing Mean Squared Error: {testing_mse:.6f}")

# Perform 10-fold cross-validation on the training data
cv_scores = cross_val_score(gnb, X, y, cv=10, scoring='neg_mean_squared_error')

# Negate the scores to get the positive MSE values
cv_mse_scores = -cv_scores

# Compute the mean of MSE across all folds
average_mse = np.mean(cv_mse_scores)
# Compute the variance of MSE across all folds
variance_mse = np.var(cv_mse_scores)

# Print the results
print(f"Average Mean Squared Error from Cross-Validation: {average_mse:.6f}")
print(f"Variance of Mean Squared Error from Cross-Validation: {variance_mse:.6f}")


# In[484]:


from sklearn.linear_model import LogisticRegression

#Logistic Regression

log_reg = LogisticRegression()
# Fit on the training data
log_reg.fit(X_train, y_train)

# Predict on the training data
y_train_pred = log_reg.predict(X_train)

# Calculate training MSE
training_mse = mean_squared_error(y_train, y_train_pred)
print(f"Training Mean Squared Error: {training_mse:.6f}")

# Predict on the test data
y_test_pred = log_reg.predict(X_test)

# Calculate testing MSE
testing_mse = mean_squared_error(y_test, y_test_pred)
print(f"Testing Mean Squared Error: {testing_mse:.6f}")

# Perform 10-fold cross-validation on the training data
cv_scores = cross_val_score(log_reg, X, y, cv=10, scoring='neg_mean_squared_error')

# Negate the scores to get the positive MSE values
cv_mse_scores = -cv_scores

# Compute the mean of MSE across all folds
average_mse = np.mean(cv_mse_scores)
# Compute the variance of MSE across all folds
variance_mse = np.var(cv_mse_scores)

# Print the results
print(f"Average Mean Squared Error from Cross-Validation: {average_mse:.6f}")
print(f"Variance of Mean Squared Error from Cross-Validation: {variance_mse:.6f}")


# In[485]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#KNN
#varibales most associated are cylinders, displacement, horsepower, weight, origin (all >.50 in correlation)
knn_dataset = auto[['mpg','cylinders','displacement','horsepower','weight','origin']]
X_selected = knn_dataset.drop(columns=['mpg'])  # Features
y_selected = knn_dataset['mpg']  # Target variable

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

k_vals = [1,3,5,7,9,11,13,15]

training_errors = []

testing_errors = []

cv_scores = []

for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit on the training data
    knn.fit(X_train, y_train)
    
    # Predict on the training data
    y_train_pred = knn.predict(X_train)
    
    # Calculate training MSE
    training_mse = mean_squared_error(y_train, y_train_pred)
    training_errors.append((k,round(training_mse, 6)))
    
    # Predict on the test data
    y_test_pred = knn.predict(X_test)
    
    # Calculate testing MSE
    testing_mse = mean_squared_error(y_test, y_test_pred)
    testing_errors.append((k,round(testing_mse, 6)))

    cv_score = cross_val_score(knn, X_selected, y_selected, cv=10, scoring='neg_mean_squared_error')
    
    cv_mse = -cv_score
    var_mse = np.var(cv_mse)
    cv_scores.append((k, cv_mse, var_mse))

cv_scores_means = []

for tup in cv_scores:
    cv_scores_means.append((tup[0], np.mean(tup[1]), tup[2]))

print(training_errors)
print(testing_errors)
print(cv_scores_means)
lowest_mean = 1
best_k = 0

for tup in cv_scores_means:
    if tup[1] < lowest_mean:
        lowest_mean = tup[1]
        best_k = tup[0]

print()
print(lowest_mean)
print(best_k)


# In[ ]:




