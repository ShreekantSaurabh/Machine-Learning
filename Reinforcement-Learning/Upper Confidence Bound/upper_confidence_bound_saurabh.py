# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
import math
N = 10000    #total no. of rounds
d = 10       #total no. of ads
ads_selected = []    #collection of ads selected in every round
#no. of times Ad i was selected upto round n
numbers_of_selections = [0] * d
#Sum of rewards of the Ad i upto round n
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(1.5 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400    #10^400 it has been done to select first 10 ads sequentially in first 10 rounds
        if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] =  numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
#Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

    
    
            
        
        

