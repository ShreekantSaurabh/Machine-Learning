#Apriori

#Data Preprocessing
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
#header = FALSE means first line is NOT the title of the columns. Its a part of data points.

#install.packages('arules')
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
#data frame is converted into Sparce format/matrix
#rm.duplicates will show the number of duplicates / triplicates

summary(dataset)
itemFrequencyPlot(dataset, topN = 100)
#top 100 frequent product purchased by the customer

#Training Apriori on the dataset
rules = apriori(data = dataset,parameter = list(support = 0.003, confidence = 0.4))

#Supoose a product is purchased 3 times a day . So in a week it has been purchased for 3*7 = 21 times
#Number of transactions in a week = 7500. Therefore, support = 21/7500 = 0.003
#Confidence is an arbitray value with default = 0.8 (ie. rule must be correct 4 out of 5 times)

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
