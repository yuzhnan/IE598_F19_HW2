#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import data, set up feature matrix and target array, train test split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('C:/Users/Thinkpad/Desktop/IE598/hw2/Treasury Squeeze test - DS1.csv', header=None)
df = df.values
y = df[1:,11]
# y=pd.DataFrame(y)
# y = y[0].map({'FALSE':False, 'TRUE':True})
# y=y.values
X = df[1:,2:10].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=33, stratify=y)


# In[4]:


# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 26)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

from sklearn.neighbors import KNeighborsClassifier
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
    

# Generate plot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[5]:


# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt_gini' with a maximum depth of 4, set 'gini' as the information criterion
dt_gini = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=1)

# Fit dt to the training set
dt_gini.fit(X_train, y_train)

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)


# In[6]:


# Import accuracy_score
from sklearn.metrics import accuracy_score

# Predict test set labels
y_pred_gini = dt_gini.predict(X_test)
y_pred_entropy = dt_entropy.predict(X_test)

# Compute test set accuracy  
accuracy_gini = accuracy_score(y_pred_gini, y_test)
accuracy_entropy = accuracy_score(y_pred_entropy, y_test)
# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)


# In[1]:


print("My name is {Yuzheng Nan}")
print("My NetID is: {ynan4}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

