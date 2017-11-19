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

#Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

#Fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2          #added new column which contains value level to the power of polynomial
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg = lm(formula = Salary ~ .,
              dataset)

#Visualising the Linear Regression result
#install.packages('ggplot2')
library(ggplot2)

ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue')+
  ggtitle('Truth or Bluff (Linear Regression)')+
  xlab('Level')+
  ylab('Salary')

#Visualising the Polynomial Regression result
#install.packages('ggplot2')
library(ggplot2)

ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red')+
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue')+
  ggtitle('Truth or Bluff (Polynomial Regression)')+
  xlab('Level')+
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
 # library(ggplot2)
 # x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
 # 
 # ggplot() +
 #   geom_point(aes(x = dataset$Level, y = dataset$Salary),
 #              colour = 'red') +
 #   geom_line(aes(x = x_grid, y = predict(poly_reg,
 #                                         newdata = data.frame(Level = x_grid,
 #                                                              Level2 = x_grid^2,
 #                                                              Level3 = x_grid^3,
 #                                                              Level4 = x_grid^4))),
 #             colour = 'blue') +
 #   ggtitle('Truth or Bluff (Polynomial Regression)') +
 #   xlab('Level') +
 #   ylab('Salary')

# Predicting a new result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new result with Polynomial Regression
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                             Level2 = 6.5^2,
                             Level3 = 6.5^3,
                             Level4 = 6.5^4))





