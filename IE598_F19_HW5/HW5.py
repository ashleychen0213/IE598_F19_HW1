import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import time

df = pd.read_csv("/Users/ashleychen/Desktop/UIUC/IE 598/HW5/hw5_treasury yield curve data.csv")

# Exploratory Data Analysis
shape = df.shape
print('Shape = {}\n'.format(shape)) 

### Identifying missing values
df.isnull().sum()

### Drop rows where all columns are NaN
drop_row = df.dropna(how='all')
new_shape = drop_row.shape
print('New Shape = {}\n'.format(new_shape))

### A table of summary statistics for each of the 30 explanatory variables
print(drop_row.iloc[:,-31:-1].describe())

### Heatmap for showing correlation for 30 explanatory variables
data = drop_row.iloc[:,-31:-1].values.T
cm =  np.corrcoef(data)
sns.set(font_scale = 1.0)
fig, ax = plt.subplots(figsize = (13,13))
col = ['SVENF01','SVENF02','SVENF03','SVENF04','SVENF05',
       'SVENF06','SVENF07','SVENF08','SVENF09','SVENF10',
       'SVENF11','SVENF12','SVENF13','SVENF14','SVENF15',
       'SVENF16','SVENF17','SVENF18','SVENF19','SVENF20',
       'SVENF21','SVENF22','SVENF23','SVENF24','SVENF25',
       'SVENF26','SVENF27','SVENF28','SVENF29','SVENF30']
hm = sns.heatmap(cm,
                cbar = True,
                square = True,
                fmt = '.2f',
                annot = True,
                annot_kws = {'size':10},
                yticklabels = col,
                xticklabels = col)
plt.show()


### Imputing missing values
imputed_data = drop_row.fillna(drop_row.mean())
imputed_shape = imputed_data.shape
print('Imputed Data Shape = {}\n'.format(imputed_shape))

### Split data into training and test sets
X = imputed_data.drop(['Date','Adj_Close'],axis = 1).values
y = imputed_data['Adj_Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.15, random_state = 42)

### Standardization
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# Perform a PCA on the Treasury Yield dataset
### Compute and Display the Explained Variance Ratio for All Components
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in
            sorted(eigen_vals, reverse = True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,31),var_exp, alpha = 0.5, align = 'center',
       label = 'individual explained variance')
plt.step(range(1,31),cum_var_exp, where = 'mid',
        label = 'cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc = 'best')
plt.show()

### PCA
pca = PCA(n_components = 3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)

### Compute and Display the Explained Variance Ratio for the 3 Components
# The Explained Variance Ratio for the 3 components:
print(pca.explained_variance_ratio_)

# The Cumulated Explained Variance Ratio for the 3 components:
print(np.cumsum(pca.explained_variance_ratio_))

plt.bar(range(1,4), pca.explained_variance_ratio_, alpha = 0.5, align = 'center',
       label = 'individual explained variance')
plt.step(range(1,4), np.cumsum(pca.explained_variance_ratio_), where = 'mid',
    label = 'cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc = 'best')
plt.show()

# The Explained Variance of the 3 Component Version:
print(pca.explained_variance_)

# The Cumulative Explained Variance of the 3 Component Version:
print(np.cumsum(pca.explained_variance_))

# Linear Regression Classifier v. SVM Classifier
### Linear Regression Classifier Model to Original Dataset
lr = linear_model.LinearRegression()
t0 = time.process_time()
lr.fit(X_train_std, y_train)
print('Training time:', round(time.process_time() - t0, 3), 's' )

lr_train_pred = lr.predict(X_train_std)
lr_test_pred = lr.predict(X_test_std)

# R2 Score:
print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, lr_train_pred),
           r2_score(y_test, lr_test_pred)))

# RMSE Score:
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, lr_train_pred)),
        sqrt(mean_squared_error(y_test, lr_test_pred))))

### Linear Classifier Model to 3 Components
new_lr = linear_model.LinearRegression()
t1 = time.process_time()
new_lr.fit(X_train_pca, y_train)
print('Training time:', round(time.process_time() - t1, 3), 's' )
lr_train_pred2 = new_lr.predict(X_train_pca)
lr_test_pred2 = new_lr.predict(X_test_pca)

# R2 Score:
print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, lr_train_pred2),
           r2_score(y_test, lr_test_pred2)))

# RMSE Score:
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, lr_train_pred2)),
        sqrt(mean_squared_error(y_test, lr_test_pred2))))

### SVM Classifier Model to Original Dataset
original_svr = svm.SVR(kernel = 'linear')
t2 = time.process_time()
original_svr.fit(X_train_std, y_train)
print('Training time:', round(time.process_time() - t2, 3), 's' )
original_train_pred = original_svr.predict(X_train_std)
original_test_pred = original_svr.predict(X_test_std)

# R2 Score:
print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, original_train_pred),
           r2_score(y_test, original_test_pred)))

# RMSE Score:
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, original_train_pred)),
        sqrt(mean_squared_error(y_test, original_test_pred))))

### SVM Classifier Model to 3 Components
new_svr = svm.SVR(kernel = 'linear')
t3 = time.process_time()
new_svr.fit(X_train_pca, y_train)
print('Training time:', round(time.process_time() - t3, 3), 's' )
new_train_pred = new_svr.predict(X_train_pca)
new_test_pred = new_svr.predict(X_test_pca)

# R2 Score:
print('R^2 train: %.3f, test: %.3f' % (
           r2_score(y_train, new_train_pred),
           r2_score(y_test, new_test_pred)))

# RMSE Score:
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, new_train_pred)),
        sqrt(mean_squared_error(y_test, new_test_pred))))

print("My name is Yu Chi Chen")
print("My NetID is: yuchicc2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
