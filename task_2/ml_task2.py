#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Christos Karapanagiotis 
"""

# Import the libraries
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

#%%

#import the data set:
higgs = pd.read_csv('/home/trolletarian/Downloads/atlas-higgs-challenge-2014-v2.csv', \
                    delimiter=",")

#select the training data set based on KaggleSet values 't'
train = higgs.loc[higgs['KaggleSet'] == 't']


#%%
# Train a neural network including all the features

input_df = train.drop(['EventId','Weight','Label','KaggleSet','KaggleWeight'], axis=1)
labels = list(train['Label'])

# Replace the 's' in the label with 1 and the background with 0 
for i in range (len(labels)):
    if labels[i]=='s':
        labels[i]=1 
    else:
        labels[i]=0 

#%%
# define the keras model
model = Sequential()
model.add(Dense(16, input_dim=30, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#optimizer
my_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, \
                    amsgrad=False)

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer=my_optimizer, metrics=['accuracy'])

# checkpoint
filepath="my_model1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

# fit the keras model on the dataset
model_history = model.fit(input_df, labels, validation_split=0.1, epochs=100, \
                          batch_size=128, callbacks=[checkpoint])      

#%%
# Drop the phi features:
input_df = input_df.drop(['PRI_jet_subleading_phi','PRI_jet_leading_phi', \
                          'PRI_met_phi','PRI_lep_phi', 'PRI_tau_phi'],axis=1)

       
# Train again:
model = Sequential()
model.add(Dense(16, input_dim=25, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#optimizer
my_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, \
                    amsgrad=False)

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer=my_optimizer, metrics=['accuracy'])

# checkpoint
filepath="my_model_drop_phi.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

# fit the keras model on the dataset
model_history = model.fit(input_df, labels,validation_split=0.1, epochs=100, \
                          batch_size=128, callbacks=[checkpoint])    
        