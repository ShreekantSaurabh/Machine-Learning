# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:46:43 2017

@author: Saurabh
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory & import data
dataset = pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

# Spliting the dataset into Training set & Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

#Fitting Regression to the dataset
# Create your regressor here

#Predicting a new result with Regression
y_pred = regressor.predict(6.5)

#Visualising the Regression result
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualising the Polynomial Regression result (better, smoother, higher resolution curve)
X_grid = np.arange(min(X), max(X), 0.1)    #To make curve better, smoother, higher resolution
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.scatter(X, Y, color = 'red')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()