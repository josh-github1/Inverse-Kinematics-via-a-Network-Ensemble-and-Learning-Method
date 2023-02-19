import os

import numpy

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras import layers

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import LeakyReLU


import keras.optimizers as Optimizers

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset with position,orientation and joint angles
dataset = numpy.loadtxt("data.csv", delimiter=",")
print(dataset.shape)
# split into input (X) and output (Y) variables
X = dataset[:10000,:7]
Y = dataset[:10000,7:]

X_test = dataset[10000:,:7]
Y_test = dataset[10000:,7:]

print(X.shape, Y.shape)

first = 7
last = 6

# Defining several networks
def model1():
     model = Sequential()
     model.add(Dense(32, input_dim=7, activation='relu'))
     model.add(Dense(64,  activation='relu'))
     model.add(Dense(128, activation='relu'))
     model.add(Dense(32, activation='relu'))
     model.add(Dense(6))
     model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
     return model



clf = KerasRegressor(build_fn=model1, epochs=500, batch_size=20, verbose=2)
history = clf.fit(X, Y)
res = clf.predict(X_test)
print(res)


score = mean_absolute_error(Y_test, res)
print(score)
# ... code
K.clear_session()