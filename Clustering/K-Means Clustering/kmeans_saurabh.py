# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory & import data
dataset = pd.read_csv("Mall_Customers.csv")

X=dataset.iloc[:,[3,4]].values

#Plot & Use the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)

#n_clusters is no. of clusters to form, init is for initializer to avoid the random initialization trap
#max_iter is max iteration of the k-means for a single run.
#n_init is no. of times k-means will run with different values of centroid
#inertia is used to calculate WCSSs

plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying k-means to the mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visualising the 5 clusters
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans ==  0,1], s = 100, c = 'red', label = 'Cluster 1 (Careful)')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans ==  1,1], s = 100, c = 'blue', label = 'Cluster 2 (Standard)')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans ==  2,1], s = 100, c = 'green', label = 'Cluster 3 (Target)')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans ==  3,1], s = 100, c = 'cyan', label = 'Cluster 4 (Careless)')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans ==  4,1], s = 100, c = 'magenta', label = 'Cluster 5 (Sensible)')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
