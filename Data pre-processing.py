# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:46:25 2020

@author: ESAMMY
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# Dealing with missing values and replacing it with the mean of that column
dataset = pd.read_csv('Data.csv.csv')
dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values
 
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,1:3])# This fits the SimpleImputer in our dataset

X[:,1:3] = imputer.transform(X[:,1:3]) # This puts the mean in the missing position
replaced_val = X[:,1:3]
replaced_val

## that is changing columns with strings to numbers for ML to understand
## firt we need to import OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# lets create columnTransformer object
# encoding independent variable
ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
                      remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding dependent variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(Y)


## Splitting dataset into Train and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler 

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#X_train = X_train.reshape(-1,1)
#Y_train = Y_train.reshape(-1,1)

regressor.fit(X_train,Y_train)


# predicting test set result
Y_pred = regressor.predict(X_test)



















































