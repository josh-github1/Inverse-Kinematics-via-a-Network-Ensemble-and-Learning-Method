import numpy as np
import tensorflow as tf
from tensorflow import keras

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
X = dataset[:1000,:7]
Y = dataset[:1000,7:]

X_test = dataset[1000:,:7]
Y_test = dataset[1000:,7:]

m1 = keras.models.load_model("m1")
m2 = keras.models.load_model("m2")
m3 = keras.models.load_model("m3")
m4 = keras.models.load_model("m4")
m5 = keras.models.load_model("m5")
m6 = keras.models.load_model("m6")


arr1a = np.load("tr1X", allow_pickle=True)
arr1b = np.load("tr1Y", allow_pickle=True)
arr2a = np.load("tr2X")
arr2b = np.load("tr2Y")
arr3a = np.load("tr3X")
arr3b = np.load("tr3Y")
arr4a = np.load("tr4X")
arr4b = np.load("tr4Y")
arr5a = np.load("tr5X")
arr5b = np.load("tr5Y")
arr6a = np.load("tr6X")
arr6b = np.load("tr6Y")



history1 = m1.fit(arr1a, arr1b, epochs=4000)
history2 = m2.fit(arr2a, arr2b, epochs=4000)
history3 = m3.fit(arr3a, arr3b, epochs=4000)
history4 = m4.fit(arr4a, arr4b, epochs=4000)
history5 = m5.fit(arr5a, arr5b, epochs=4000)
history6 = m6.fit(arr6a, arr6b, epochs=4000)


print(history1.history.keys())

plt.plot(history1.history['accuracy'], label="Model 1")
plt.plot(history2.history['accuracy'], label="Model 2")
plt.plot(history3.history['accuracy'], label="Model 3")
plt.plot(history4.history['accuracy'], label="Model 4")
plt.plot(history5.history['accuracy'], label="Model 5")
plt.plot(history6.history['accuracy'], label="Model 6")

plt.title('Model Accuracy for the first extracted set on 1000 data points')
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

plt.title('Model Loss for the 1st extracted set on 1000 data points')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left', borderaxespad=0.)
plt.show()

m1.save("m1a")
m2.save("m2a")
m3.save("m3a")
m4.save("m4a")
m5.save("m5a")
m6.save("m6a")


