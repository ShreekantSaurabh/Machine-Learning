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

#Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)

#Predicting a new result with Decision Tree Regression
y_pred = regressor.predict(6.5)

#Visualising the Decision Tree Regression result (better, smoother, higher resolution curve)
#Note : Decision Tree is non-continuous model, so visualisation should be on higher resolution
X_grid = np.arange(min(X), max(X), 0.01)    #To make curve better, smoother, higher resolution
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.scatter(X, Y, color = 'red')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()