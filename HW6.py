# Homework 6
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

### Read in the ccdefault dataset
df = pd.read_csv("/Users/ashleychen/Desktop/UIUC/IE 598/HW6/ccdefault.csv")

## Part 1: Random test train splits

 ### Split the data into train set and test set, and print out the accuracy scores
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
train_accuracy = []
test_accuracy = []

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = i, stratify = y)
    dt = DecisionTreeClassifier(random_state = 1, criterion = 'gini', max_depth = 6 )
    dt.fit(X_train, y_train)
    ytrain_pred = dt.predict(X_train)
    ytest_pred = dt.predict(X_test)
    
    print('For sample '+ str(i) + ": \n Train Accuracy: %.6f; Test accuracy: %.6f" % (accuracy_score(y_train, ytrain_pred), accuracy_score(y_test,ytest_pred)))
    train_accuracy.append(accuracy_score(y_train, ytrain_pred))
    test_accuracy.append(accuracy_score(y_test, ytest_pred))

### Calculate the mean and standard deviation on the set of scores
print("Train Set:\n Mean: " + str(np.mean(train_accuracy)) + "; Standard Deviation: " + str(np.std(train_accuracy)))
print("Test Set:\n Mean: " + str(np.mean(test_accuracy)) + "; Standard Deviation: " + str(np.std(test_accuracy)))

## Part 2: Cross validation
### Report the individual fold accuracy scores
kfold = StratifiedKFold(n_splits = 10, random_state = 1).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    dt.fit(X_train[train], y_train[train])
    score = dt.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Acc: %.33f' % (k+1, score))

### Calculate the mean and standard deviation on the test set of scores
print("Test set:\n Mean: " + str(np.mean(scores))+ "; Standard deviation: " + str(np.std(scores)))

print("My name is Yu Chi Chen")
print("My NetID is: yuchicc2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

