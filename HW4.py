#Homework 4
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv('/Users/ashleychen/Desktop/UIUC/IE 598/HW4/Dataset/housing2.csv', 
                 header = 0)


## EDA
shape = df.shape
print('Shape = {}\n'.format(shape)) 


### Determining the Nature of Attributes
df.dtypes


### A table of summary statistics for each of the 13 explanatory variables
print(df.iloc[:,-14:-1].describe())

### Heatmap for showing correlation for 13 explanatory variables
data = df.iloc[:,-14:-1].values.T
cm =  np.corrcoef(data)
sns.set(font_scale=1.0)
fig, ax = plt.subplots(figsize=(12,12))
col = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
hm = sns.heatmap(cm,
                cbar=True,
                square=True,
                fmt='.2f',
                annot=True,
                annot_kws={'size':10},
                yticklabels=col,
                xticklabels=col)
plt.show()


### Identifying missing values
df.isnull().sum()
# Since we identify there are 54 missing values for MEDV, we should impute missing values.

### Imputing missing values
imputed_data = df.fillna(df.mean())
print(imputed_data)

### Split data into training and test sets
X = imputed_data.drop(['MEDV'],axis = 1).values
y = imputed_data['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

## Linear Regressions
slr = linear_model.LinearRegression()
slr.fit(X_train,y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
print(slr.coef_)
print(slr.intercept_)

### Plot the residual errors
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'steelblue', marker = 's', edgecolor = 'white',
           label = 'Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
            c = 'limegreen', marker = 's', edgecolor = 'white',
            label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()

### Calculate MSE and R2
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, y_train_pred),
           r2_score(y_test, y_test_pred)))
    
## Ridge Regression

# For alpha = 1,
ridge = Ridge(alpha = 1, normalize = True)
ridge.fit(X_train, y_train)
print(ridge.coef_)
print(ridge.intercept_)

# Plot the residual errors
ridge_train_pred = ridge.predict(X_train)
ridge_test_pred = ridge.predict(X_test)
plt.scatter(ridge_train_pred, ridge_train_pred - y_train, c = 'steelblue', marker = 's', edgecolor = 'white',
           label = 'Training data')
plt.scatter(ridge_test_pred, ridge_test_pred - y_test,
            c = 'limegreen', marker = 's', edgecolor = 'white',
            label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()

# Calculate MSE and R2
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, ridge_train_pred),
        mean_squared_error(y_test, ridge_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, ridge_train_pred),
           r2_score(y_test, ridge_test_pred)))
    


# For alpha = 0.1
ridge_1 = Ridge(alpha = 0.1, normalize = True)
ridge_1.fit(X_train, y_train)
print(ridge_1.coef_)
print(ridge_1.intercept_)

# Plot the residual errors
ridge_1_train_pred = ridge_1.predict(X_train)
ridge_1_test_pred = ridge_1.predict(X_test)
plt.scatter(ridge_1_train_pred, ridge_1_train_pred - y_train, c = 'steelblue', marker = 's', edgecolor = 'white',
           label = 'Training data')
plt.scatter(ridge_1_test_pred, ridge_1_test_pred - y_test,
            c = 'limegreen', marker = 's', edgecolor = 'white',
            label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()

# Calculate MSE and R2
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, ridge_1_train_pred),
        mean_squared_error(y_test, ridge_1_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, ridge_1_train_pred),
           r2_score(y_test, ridge_1_test_pred)))


# For alpha = 2
ridge_2 = Ridge(alpha = 2, normalize = True)
ridge_2.fit(X_train, y_train)
print(ridge_2.coef_)
print(ridge_2.intercept_)

# Plot the residual errors
ridge_2_train_pred = ridge_2.predict(X_train)
ridge_2_test_pred = ridge_2.predict(X_test)
plt.scatter(ridge_2_train_pred, ridge_2_train_pred - y_train, c = 'steelblue', marker = 's', edgecolor = 'white',
           label = 'Training data')
plt.scatter(ridge_2_test_pred, ridge_2_test_pred - y_test,
            c = 'limegreen', marker = 's', edgecolor = 'white',
            label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()

# Calculate MSE and R2
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, ridge_2_train_pred),
        mean_squared_error(y_test, ridge_2_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, ridge_2_train_pred),
           r2_score(y_test, ridge_2_test_pred)))


## Lasso Regression
# For alpha = 1,
lasso = Lasso(alpha = 1.0, normalize = True)
lasso.fit(X_train, y_train)
print(lasso.coef_)
print(lasso.intercept_)

# Plot the residual errors
lasso_train_pred = lasso.predict(X_train)
lasso_test_pred = lasso.predict(X_test)
plt.scatter(lasso_train_pred, lasso_train_pred - y_train, c = 'steelblue', marker = 's', edgecolor = 'white',
           label = 'Training data')
plt.scatter(lasso_test_pred, lasso_test_pred - y_test,
            c = 'limegreen', marker = 's', edgecolor = 'white',
            label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw = 2)
plt.xlim([-10,50])
plt.show()

# Calculate MSE and R2
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, lasso_train_pred),
        mean_squared_error(y_test, lasso_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, lasso_train_pred),
           r2_score(y_test, lasso_test_pred)))

# For alpha = 0.1,
lasso_1 = Lasso(alpha = 0.1, normalize = True)
lasso_1.fit(X_train, y_train)
print(lasso_1.coef_)
print(lasso_1.intercept_)

# Plot the residual errors
lasso_1_train_pred = lasso_1.predict(X_train)
lasso_1_test_pred = lasso_1.predict(X_test)
plt.scatter(lasso_1_train_pred, lasso_1_train_pred - y_train, c = 'steelblue', marker = 's', edgecolor = 'white',
           label = 'Training data')
plt.scatter(lasso_1_test_pred, lasso_1_test_pred - y_test,
            c = 'limegreen', marker = 's', edgecolor = 'white',
            label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()

# Calculate MSE and R2
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, lasso_1_train_pred),
        mean_squared_error(y_test, lasso_1_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, lasso_1_train_pred),
           r2_score(y_test, lasso_1_test_pred)))

# For alpha = 0.01,
lasso_2 = Lasso(alpha = 0.01, normalize = True)
lasso_2.fit(X_train, y_train)
print(lasso_2.coef_)
print(lasso_2.intercept_)

# Plot the residual errors
lasso_2_train_pred = lasso_2.predict(X_train)
lasso_2_test_pred = lasso_2.predict(X_test)
plt.scatter(lasso_2_train_pred, lasso_2_train_pred - y_train, c = 'steelblue', marker = 's', edgecolor = 'white',
           label = 'Training data')
plt.scatter(lasso_2_test_pred, lasso_2_test_pred - y_test,
            c = 'limegreen', marker = 's', edgecolor = 'white',
            label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()

# Calculate MSE and R2
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, lasso_2_train_pred),
        mean_squared_error(y_test, lasso_2_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, lasso_2_train_pred),
           r2_score(y_test, lasso_2_test_pred)))

print("My name is Yu Chi Chen")
print("My NetID is: yuchicc2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
