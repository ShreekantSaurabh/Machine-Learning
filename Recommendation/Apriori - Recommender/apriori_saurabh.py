# Apriori
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
# header = None means first line is NOT the title of the columns. Its a part of data points.

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.4, min_lift = 3, min_length = 3)

#Supoose a product is purchased 3 times a day . So in a week it has been purchased for 3*7 = 21 times
#Number of transactions in a week = 7500. Therefore, support = 21/7500 = 0.003
#Confidence is an arbitray value with default = 0.8 (ie. rule must be correct 4 out of 5 times)


# Visualising the results
results = list(rules)