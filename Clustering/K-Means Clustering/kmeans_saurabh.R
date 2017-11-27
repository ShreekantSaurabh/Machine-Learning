# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Plot & Use the elbow method to find the optimal number of clusters
set.seed(6)
WCSS = vector()
for (i in 1:10)
  WCSS[i] = sum(kmeans(X, i)$withinss)
#kmeans has attribute withinss which calculate WCSS

plot(1:10, WCSS, type = "b", main = paste('Clusters of clients'),
     xlab = "Number of clusters", ylab =  "WCSS")
#type = "b" is for both lines and dots

# Applying K-Means to the dataset
set.seed(29)
kmeans = kmeans(X, centers = 5, iter.max = 300, nstart = 10)
y_kmeans = kmeans$cluster
#centers is no. of clusters to form
#iter.max is max iteration of the k-means for a single run.
#nstart is no. of times k-means will run with different values of centroid

# Visualising the clusters
#install.packages('cluster')
library(cluster)
clusplot(X,
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')