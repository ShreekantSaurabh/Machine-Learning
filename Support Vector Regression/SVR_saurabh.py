# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory & import data
dataset = pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values.reshape(-1, 1)

# Spliting the dataset into Training set & Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
#Note: SVR does not do feature scaling by default
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#Predicting a new result with Regression
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualising the SVR result
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the SVR result (better, smoother, higher resolution curve)
X_grid = np.arange(min(X), max(X), 0.1)    #To make curve better, smoother, higher resolution
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.scatter(X, Y, color = 'red')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()