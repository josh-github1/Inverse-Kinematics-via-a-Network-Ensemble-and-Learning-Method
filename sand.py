# Evaluating all models
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

from mpl_toolkits import mplot3d

# load dataset with position,orientation and joint angles
dataset = numpy.loadtxt("data.csv", delimiter=",")
print(dataset.shape)
# split into input (X) and output (Y) variables
X = dataset[:1000,:7]
Y = dataset[:1000,7:]
X_test = dataset[1000:,:7]
Y_test = dataset[1000:,7:]


plt.plot(X[:1000, 1], X[:1000, 2])

fig = plt.figure()
ax = plt.axes(projection='3d')

plt.show()

ax.scatter3D(Y[:1000, 1], Y[:1000, 2], Y[:1000, 3], cmap='Greens')