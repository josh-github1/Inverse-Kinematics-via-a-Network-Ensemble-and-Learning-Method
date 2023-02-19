# 3RD SET ON TRAINING ON ORIGINAL DATASET


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os

import math
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
np.random.seed(seed)

# load dataset with position,orientation and joint angles
dataset = np.loadtxt("data.csv", delimiter=",")
print(dataset.shape)

# split into input (X) and output (Y) variables
X = dataset[:10000,:7]
Y = dataset[:10000,7:]

X_test = dataset[10000:,:7]
Y_test = dataset[10000:,7:]

m1 = keras.models.load_model("m1c")
m2 = keras.models.load_model("m2c")
m3 = keras.models.load_model("m3c")
m4 = keras.models.load_model("m4c")
m5 = keras.models.load_model("m5c")
m6 = keras.models.load_model("m6c")


history1 = m1.fit(X, Y, epochs=1000)
history2 = m2.fit(X, Y, epochs=1000)
history3 = m3.fit(X, Y, epochs=1000)
history4 = m4.fit(X, Y, epochs=1000)
history5 = m5.fit(X, Y, epochs=1000)
history6 = m6.fit(X, Y, epochs=1000)


print(history1.history.keys())

plt.plot(history1.history['accuracy'], label="Model 1")
plt.plot(history2.history['accuracy'], label="Model 2")
plt.plot(history3.history['accuracy'], label="Model 3")
plt.plot(history4.history['accuracy'], label="Model 4")
plt.plot(history5.history['accuracy'], label="Model 5")
plt.plot(history6.history['accuracy'], label="Model 6")

plt.title('Model Accuracy on 1000 data points, set 3')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left', borderaxespad=0.)

plt.show()

# summarize history for loss
plt.plot(history1.history['loss'], label="Model 1")
plt.plot(history2.history['loss'], label="Model 2")
plt.plot(history3.history['loss'], label="Model 3")
plt.plot(history4.history['loss'], label="Model 4")
plt.plot(history5.history['loss'], label="Model 5")
plt.plot(history6.history['loss'], label="Model 6")

plt.title('Model Loss on 1000 data points, set 3')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left', borderaxespad=0.)
plt.show()


m1.save("m1d")
m2.save("m2d")
m3.save("m3d")
m4.save("m4d")
m5.save("m5d")
m6.save("m6d")
