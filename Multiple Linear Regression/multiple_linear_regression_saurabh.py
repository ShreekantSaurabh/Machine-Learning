# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#Set working directory & import data
dataset = pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

#Encoding categorical data
#Encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap , it is automatically taken care by model so no need to specify
#X = X[:,1:]

# Spliting the dataset into Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set result
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

#Adding X0 column which has all elements as 1
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
#All possible Predictor
X_opt = X[:,[0,1,2,3,4,5]]
#Fit the model with all possible predictor
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing the predictor with highest P-value
#All Predictor except X2 (pvalue = 0.99)
X_opt = X[:,[0,1,3,4,5]]
#Fit the model with all possible predictor
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing the predictor with highest P-value
#All Predictor except X1 (pvalue = 0.94)
X_opt = X[:,[0,3,4,5]]
#Fit the model with all possible predictor
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing the predictor with highest P-value
#All Predictor except X2 (pvalue = 0.6)
X_opt = X[:,[0,3,5]]
#Fit the model with all possible predictor
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#Removing the predictor with highest P-value
#All Predictor except X2 (pvalue = 0.06)
X_opt = X[:,[0,3]]
#Fit the model with all possible predictor
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
