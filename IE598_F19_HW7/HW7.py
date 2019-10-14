# Homework 7
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time

### Read in the ccdefault dataset
df = pd.read_csv("/Users/ashleychen/Desktop/UIUC/IE 598/HW6/ccdefault.csv")

### Split the data into train set and test set
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1, stratify = y)

### Part 1: Fit a random forest model for different n_estimators
size = [10, 50, 100, 200]
for i in size:
    forest = RandomForestClassifier(criterion='gini', n_estimators = i, random_state = 1)
    t0 = time.process_time()
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    scores = cross_val_score(estimator = forest,
                             X = X_train,
                             y = y_train,
                             cv = 10)
    mean = scores.mean()
    std = scores.std()
    print('For n_estimators = '+ str(i) + ': \n Accuracy: %0.6f (+/- %0.6f)'% 
          (mean, std))
    print(' Computation time:', round(time.process_time() - t0, 3), 's' )

### Part 2: Random forest feature importance
forest = RandomForestClassifier(n_estimators = 200, random_state=1)
feat_labels = df.columns[0:]
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1,
                            30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
    
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is Yu Chi Chen")
print("My NetID is: yuchicc2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am     not in violation.")

