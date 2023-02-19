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


# load dataset with position,orientation and joint angles
dataset = numpy.loadtxt("data.csv", delimiter=",")
print(dataset.shape)
# split into input (X) and output (Y) variables
X = dataset[:1000,:7]
Y = dataset[:1000,7:]
X_test = dataset[1000:,:7]
Y_test = dataset[1000:,7:]

basem1 = keras.models.load_model("baseM1")
basem4 = keras.models.load_model("baseM4")

m1 = keras.models.load_model("m1d")
m2 = keras.models.load_model("m2d")
m3 = keras.models.load_model("m3d")
m4 = keras.models.load_model("m4d")
m5 = keras.models.load_model("m5d")
m6 = keras.models.load_model("m6d")



res1 = basem1.predict(X_test)
mape = tf.keras.losses.MeanAbsolutePercentageError()
q = mape(Y_test, res1).numpy()
print("Percentage error for base m1: ")
print(q)

res2 = basem4.predict(X_test)
mape = tf.keras.losses.MeanAbsolutePercentageError()
q = mape(Y_test, res1).numpy()
print("percentage error for base m4: ")
print(q)

res1 = m4.predict(X_test)
sc3 = mean_squared_error(Y_test, res1)
mape = tf.keras.losses.mean_squared_error(Y_test, res1)
# q = mape(Y_test, res1).numpy()
print("MSE for best model of NN committee machine, m4: ")
print(sc3)

res2 = m4.predict(X_test)
mape = tf.keras.losses.MeanAbsolutePercentageError()
q = mape(Y_test, res2).numpy()
print("percentage error for m4: ")
print(q)

print("Evaluate on test data")
results = basem1.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

print("Evaluate on test data")
results = basem4.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

print("Evaluate on test data")
results = m1.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

print("Evaluate on test data")
results = m2.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

print("Evaluate on test data")
results = m3.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

print("Evaluate on test data")
results = m4.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

print("Evaluate on test data")
results = m5.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

print("Evaluate on test data")
results = m6.evaluate(X_test, Y_test, batch_size=128)
print("test loss, test acc:", results)

