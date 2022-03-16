# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:22:30 2020

@author: ESAMMY
"""


#importing libraring
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Customer List.csv')
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values

# Splitting dataset into Train and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0,)

# feature scaling
# we do feature scaling when a machine learning algorithm useing euclidian distance


# Fitting Random forest to training sets
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

# predicting test set results
Y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# not we can calculate the accuracy of the model 
# -- Accuracy rate = Correct/Total
# Accuracy rate = 89/100 = 89% 

# -- Error rate = Wrong/Total
# Error rate = 11/100 = 11%







