#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Christos Karapanagiotis
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from keras.models import load_model

#%%

#import the data set:
higgs = pd.read_csv('atlas-higgs-challenge-2014-v2.csv', delimiter=",")

#select the test data set based on KaggleSet values 'b'
test = higgs.loc[higgs['KaggleSet'] == 'b']


# Split the column into features and labels:

test_df = test.drop(['EventId','Weight','Label','KaggleSet','KaggleWeight'], axis=1)
labels = list(test['Label'])

# Replace the 's' in the label with 1 and the background with 0 
for i in range (len(labels)):
    if labels[i]=='s':
        labels[i]=1 
    else:
        labels[i]=0 

#%%        

# Load the model:
        
filepath="my_model1.hdf5"
model = load_model(filepath)

# get the predictions

predictions = model.predict(test_df)

#%%
weights_test = test['KaggleWeight']

#Draw a histogram:
plt.figure
plt.hist(predictions, bins=49, weights = weights_test, edgecolor='black', linewidth=0.2)
plt.title('Number of events for the new score-feature')
plt.ylabel('# Events')
plt.xlabel('Score_feature')
plt.yscale('log')
plt.show

# Draw a Density plot:
flattened_pred = []
for x in predictions:
    for y in x:
        flattened_pred.append(y)
        
kernel = stats.gaussian_kde(flattened_pred)

positions = np.linspace(-0.2, 1.2, 1000)

plt.figure() 
plt.plot(positions, kernel(positions))
plt.title('Density plot of the new score-feature')
plt.ylabel('# Events')
plt.xlabel('Score feature')
plt.show()

#%%

#Significance vs Threshold

def significance(data, labels, threshold, weights):
    '''
    Parameters:
        data: the predicted data
        labels: the true labels of the events (given)
        threshold: the threshold which defines the signal region
        weights: the weights of the events (given) / must be in a list
    
    Returns:
        The significance which is ratio of the number of signal which has been 
        predicted correctly (true positive) and the number of signal which has 
        been predicted incorrectly (false postive).
        
    '''
    s = 0
    b = 0
    for i in range(len(data)):
        
        if data[i] >= threshold and labels[i] == 1:
            s += weights[i] # true positive
            
        if data[i] >= threshold and labels[i] == 0:
            b += weights[i] # false positive
    
    return s/np.sqrt(b) # significance

thres_val = np.linspace(0,1,100) # Define the threshold values
sign_list = list() #list of significance values for thresholds in range [0,1]

for thres in thres_val:
    sign_list.append(significance(predictions, labels, thres, list(weights_test)))
  
    
# Plot Significance vs Threshold:
plt.figure()    
plt.plot(thres_val, sign_list)
plt.title('Significance vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Significance')
plt.show()


#maximum significance:
max_sign = max(sign_list)
print (max_sign)

#Threshold that gives the maximum significance:
best_threshold = thres_val[sign_list.index(max_sign)]
print (best_threshold)