#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/ashleychen/Desktop/UIUC/IE 598/HW2/Treasury Squeeze test.csv', header = None)
df.head()


# In[2]:


data = df.to_numpy()
X = data[1:,2:-1]
y = data[1:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 33)


# In[3]:


k_range = range(1, 200)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))


# In[4]:


#Create plot to see which k may have a higher accuracy score
plt.plot(np.arange(1,200),scores)
#Find out in the range(1,200), k=116 has the highest accuracy score
np.argmax(scores)
#The accuracy score that k has is about 0.7056
scores[116]


# In[5]:


gini_tree = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=1)
gini_tree.fit(X_train, y_train)
gini_pred = gini_tree.predict(X_test)
accuracy_score(y_test, gini_pred)


# In[6]:


entropy_tree = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=1)
entropy_tree.fit(X_train, y_train)
entropy_pred = entropy_tree.predict(X_test)
accuracy_score(y_test,entropy_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




