#Importing data
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Spliting the dataset into Training set & Test set
# Spliting the dataset into Training set & Test set
#We dont have enough data to split it into train and test set
#install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 0.8)
# training_set = subset(dataset, split == T)
# test_set = subset(dataset, split == F)


#Fitting Regression Model to the dataset
# Create your regressor here

# Predicting a new result with Regression Model
y_pred = predict(regressor, data.frame(Level = 6.5))


#Visualising the Regression Model result
#install.packages('ggplot2')
library(ggplot2)

ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue')+
  ggtitle('Truth or Bluff (Regression Model)')+
  xlab('Level')+
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')