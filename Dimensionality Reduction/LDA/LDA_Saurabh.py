# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory & import data
dataset = pd.read_csv("Wine.csv")
X=dataset.iloc[:, 0:13].values
Y=dataset.iloc[:,13].values

# Spliting the dataset into Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling (to make the features on same scale)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, Y_train)
X_test = lda.transform(X_test)

#n_components is no. of Linear Disriminant ie. no. features we want to keep which will explain max. variance
#First keep the n_components = None, after looking at the explained_variance_ratio_ we change it to desired value
#fit_transform(X_train, Y_train) in case of Supervised model LDA and fit_transform(X_train) incase of unsupervised model PCA

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

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

plt.contour(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())    #x & y coordinates range
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

plt.title('Logistic Regression (Training Set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

#Visualising the Test set result
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test     #local assignment of 'train' into 'set' helps in reusability 
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step =0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step =0.01))
#meshgrid is used to specify the pixel/coordinate. start & stop is the starting & ending pixel value of the grid with increment of resolution size(step). 
#min-1 & max+1 is done to avoid the data points touching the start & end line of the grid.
               
plt.contour(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())    #x & y coordinates range
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

plt.title('Logistic Regression (Test Set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()
