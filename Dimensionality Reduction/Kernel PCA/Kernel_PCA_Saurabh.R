#Importing data
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[ , 3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Spliting the dataset into Training set & Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == T)
test_set = subset(dataset, split == F)

#Feature Scaling
training_set[ , 1:2] = scale(training_set[ , 1:2])
test_set[ , 1:2] = scale(test_set[ , 1:2])

# Applying Kernel PCA
#install.packages('kernlab')
library(kernlab)
kpca = kpca(~., data = training_set[-3], kernel = 'rbfdot', features = 2)
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$Purchased = training_set$Purchased
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased

#Fitting Logistic Regression to the Training set
classifier = glm(formula = Purchased ~ ., family = binomial(), data = training_set_pca)

#Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set_pca[,-3])
y_pred = ifelse(prob_pred >0.5, 1, 0)
#if else statement if condition true then output is 1, else 0

#Making Confusion Matrix to evaluate the prediction
cm = table(test_set_pca[, 3], y_pred)

# Visualising the Training set results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
#expand.grid is used to specify pixel/coordinate. starting & ending pixel of the grid with increment of resolution size(by).
#min-1 & max+1 is done to avoid the data points touching the start & end line of the grid.
#set[, 1] is age column and set[, 2] is Salary column

colnames(grid_set) = c('V1', 'V2')

prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))    #x & y coordinates range . set[, -3] is for Age and Salary 

contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))    #background color
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))                 #data point color

# Visualising the Test set results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
#expand.grid is used to specify pixel/coordinate. starting & ending pixel of the grid with increment of resolution size(by).
#min-1 & max+1 is done to avoid the data points touching the start & end line of the grid.
#set[, 1] is age column and set[, 2] is Salary column

colnames(grid_set) = c('V1', 'V2')

prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))    #x & y coordinates range . set[, -3] is for Age and Salary 

contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))    #background color
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))                 #data point color