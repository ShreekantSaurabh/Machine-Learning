# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory & import data
dataset = pd.read_csv("Social_Network_Ads.csv")
X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values

# Spliting the dataset into Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling (to make the features on same scale)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Classifier to the Training set
#Create your classifier here

#Predicting the Test set result
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix to evaluate the prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

#Visualising the Training set result
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train     #local assignment of 'train' into 'set' helps in reusability
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step =0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step =0.01))
#meshgrid is used to specify the pixel/coordinate. start & stop is the starting & ending pixel value of the grid with increment of resolution size(step). 
#min-1 & max+1 is done to avoid the data points touching the start & end line of the grid.
#X_set[:,0] is age column and X_set[:,1] is Salary column
#X1 is age, X2 is Salary column
            
plt.contour(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())            #x & y coordinates range
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Classifier (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualising the Test set result
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test       #local assignment of 'train' into 'set' helps in reusability
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step =0.01),
                     np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step =0.01))
#meshgrid is used to specify the pixel/coordinate. start & stop is the starting & ending pixel value of the grid with increment of resolution size(step). 
#min-1 & max+1 is done to avoid the data points touching the start & end line of the grid.
#X_set[:,0] is age column and X_set[:,1] is Salary column
#X1 is age, X2 is Salary column

plt.contour(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())            #x & y coordinates range
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Classifier (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()