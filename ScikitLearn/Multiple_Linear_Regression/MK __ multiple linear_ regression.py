# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:41:54 2017

@author: Mostafa
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features =[3] )
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

y_pred = regressor.predict(X_test)


#digging into the backward elimination
import statsmodels.formula.api as sm
X = np.append(np.ones((50,1)).astype(int), values = X , axis = 1)
X_opt = X[: , [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

#X3 has the highest P value so we are going to remove it and re fit our model

X_opt = X[: , [0,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,3,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,3]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()





























