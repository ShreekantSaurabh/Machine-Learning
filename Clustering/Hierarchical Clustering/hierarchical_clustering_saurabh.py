# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory & import data
dataset = pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values
#column 3 & 4 are annual income and spending score

#Plot & Use the Dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
#dendrogram plots the hierarchical clustering as a dendrogram.
#method = 'ward' ward method minimize the variance within each cluster

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Fitting Hierarchical Clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#n_clusters is optimal no. of clusters to be formed
#affinity is Metric used to compute the linkage. If linkage is “ward”, only “euclidean” is accepted.
#linkage = 'ward' ward ward minimizes the variance of the clusters being merged.
#The linkage criterion determines which distance to use between sets of observation to merge the pairs of cluster.

#Visualising the 5 clusters (only for 2-dimensional features)
plt.scatter(X[y_hc == 0,0], X[y_hc ==  0,1], s = 100, c = 'red', label = 'Cluster 1 (Careful)')
plt.scatter(X[y_hc == 1,0], X[y_hc ==  1,1], s = 100, c = 'blue', label = 'Cluster 2 (Standard)')
plt.scatter(X[y_hc == 2,0], X[y_hc ==  2,1], s = 100, c = 'green', label = 'Cluster 3 (Target)')
plt.scatter(X[y_hc == 3,0], X[y_hc ==  3,1], s = 100, c = 'cyan', label = 'Cluster 4 (Careless)')
plt.scatter(X[y_hc == 4,0], X[y_hc ==  4,1], s = 100, c = 'magenta', label = 'Cluster 5 (Sensible)')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



