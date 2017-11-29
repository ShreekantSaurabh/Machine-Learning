#Importing data
dataset = read.csv('Wine.csv')

# Spliting the dataset into Training set & Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == T)
test_set = subset(dataset, split == F)

#Feature Scaling
training_set[ , -14] = scale(training_set[ , -14])
test_set[ , -14] = scale(test_set[ , -14])

#Applying PCA
#install.packages('MASS')
library(MASS)
lda = lda(formula = Customer_Segment ~ ., data = training_set)

training_set = as.data.frame(predict(lda, training_set))
#changing the matrix to data frame, since predict function needs dataframe
#Changing the column order/position
training_set = training_set[c(5,6,1)]

test_set = as.data.frame(predict(lda, test_set))
#changing the matrix to data frame, since predict function needs dataframe
#Changing the column order/position
test_set = test_set[c(5,6,1)]

#Fitting SVM Classifier to the Training set
#install.packages('e1071')
library(e1071)
classifier = svm(formula = class ~ ., data = training_set, 
                 type = 'C-classification', kernel = 'linear')

#Predicting the Test set results 
y_pred = predict(classifier, newdata = test_set[,-3])

#Making Confusion Matrix to evaluate the prediction
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
#expand.grid is used to specify pixel/coordinate. starting & ending pixel of the grid with increment of resolution size(by).
#min-1 & max+1 is done to avoid the data points touching the start & end line of the grid.

colnames(grid_set) = c('x.LD1', 'x.LD2')

y_grid = predict(classifier, newdata = grid_set)

plot(set[, -3],
     main = 'SVM Classifier (Training set)',
     xlab = 'x.LD1', ylab = 'x.LD2',
     xlim = range(X1), ylim = range(X2))    #x & y coordinates range . set[, -3] is for Age and Salary 

contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))    #background color
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[,3] == 1, 'green4', 'red3')))                 #data point color

# Visualising the Test set results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
#expand.grid is used to specify pixel/coordinate. starting & ending pixel of the grid with increment of resolution size(by).
#min-1 & max+1 is done to avoid the data points touching the start & end line of the grid.

colnames(grid_set) = c('x.LD1', 'Px.LD2')

y_grid = predict(classifier, newdata = grid_set)

plot(set[, -3],
     main = 'SVM Classifier (Test set)',
     xlab = 'x.LD1', ylab = 'x.LD2',
     xlim = range(X1), ylim = range(X2))    #x & y coordinates range . set[, -3] is for Age and Salary 

contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))    #background color
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[,3] == 1, 'green4', 'red3')))                 #data point color
