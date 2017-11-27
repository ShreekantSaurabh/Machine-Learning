# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#Set working directory & import data
#os.chdir("D:\\Machine Learning\\Simple_Linear_Regression")
dataset = pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values                #iloc is index based slicing in Pandas, iloc[:,:-1] means all rows, all columns except last one.
Y=dataset.iloc[:,1].values                  #iloc[:,1] means all rows, 1st column (in python, coloumn starts from 0)

# Spliting the dataset into Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
