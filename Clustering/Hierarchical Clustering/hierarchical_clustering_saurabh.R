# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Plot & Use the Dendrogram to find the optimal number of clusters
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram, main = paste('Clusters of clients'),
     xlab = "Customers", ylab =  "Euclidean distances")

#dendrogram plots the hierarchical clustering as a dendrogram.
#method = 'ward.D' minimize the variance within each cluster

#Fitting Hierarchical Clustering to the mall dataset
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

#Visualising the 5 clusters (only for 2-dimensional features)
#install.packages('cluster')
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')