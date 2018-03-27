# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:15:08 2018

@author: Alessandro
"""

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation,LSTM
from keras.optimizers import SGD
import numpy as np
import scipy.io as sci
from features2XY import features2XY


train_data_dir = 'features_VENICE.mat'
test_data_dir = 'features_VENICE.mat'

print('load training data...')
train_data = sci.loadmat(train_data_dir)
X_train, Y_train = features2XY(train_data['features'][0], train_data['counts'][0])
test_data = sci.loadmat(test_data_dir)
X_test, Y_test = features2XY(test_data['features'][0], test_data['counts'][0])

# define fully connected regress network
model = Sequential()
model.add(Dense(100, input_dim=1000))
model.add(Dense(100))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))

print(model.summary())
#exit() 

model.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])
model.fit(X_train, Y_train, epochs=15, batch_size=1000)
result = model.evaluate(X_test, Y_test, batch_size=1000, verbose=1, sample_weight=None)
print(result)

predictions = model.predict(X_test, batch_size=1000, verbose=0)
sci.savemat('predictions_VEN.mat', {'predictions':predictions})

# serialize model to JSON
model_json = model.to_json()
with open("model_VEN.json", "w") as json_file: json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_VEN.h5")
print("Saved model to disk")

