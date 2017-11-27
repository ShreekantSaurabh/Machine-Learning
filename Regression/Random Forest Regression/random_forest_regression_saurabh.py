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

#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0 )
regressor.fit(X, Y)
#Note : n_estimators is no. of trees in Random Forest
# random_state is equivalent to set seed in R

#Predicting a new result with Random Forest Regression
y_pred = regressor.predict(6.5)

#Visualising the Random Forest Regression result (better, smoother, higher resolution curve)
X_grid = np.arange(min(X), max(X), 0.01)    #To make curve better, smoother, higher resolution
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.scatter(X, Y, color = 'red')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()