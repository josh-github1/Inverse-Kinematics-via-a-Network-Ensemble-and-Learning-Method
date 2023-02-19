# 3RD SET ON TRAINING ON EXTRACTED DATASET (Should I skip this? No data was extracted.)
# This is the one that I didn't run

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

arr1a = np.load("3tr1X", allow_pickle=True)
arr1b = np.load("3tr1Y", allow_pickle=True)
arr2a = np.load("3tr2X", allow_pickle=True)
arr2b = np.load("3tr2Y", allow_pickle=True)
arr3a = np.load("3tr3X", allow_pickle=True)
arr3b = np.load("3tr3Y", allow_pickle=True)
arr4a = np.load("3tr4X", allow_pickle=True)
arr4b = np.load("3tr4Y", allow_pickle=True)
arr5a = np.load("3tr5X", allow_pickle=True)
arr5b = np.load("3tr5Y", allow_pickle=True)
# arr6a = np.load("3tr6X", allow_pickle=True)
# arr6b = np.load("3tr6Y", allow_pickle=True)

arr1a = np.array(arr1a, dtype=np.float)
arr1b = np.array(arr1b, dtype=np.float)


m1 = keras.models.load_model("m1d")
m2 = keras.models.load_model("m2d")
m3 = keras.models.load_model("m3d")
m4 = keras.models.load_model("m4d")
m5 = keras.models.load_model("m5d")
m6 = keras.models.load_model("m6d")


history1 = m1.fit(arr1a, arr1b, epochs=4000)
history2 = m2.fit(arr2a, arr2b, epochs=4000)
history3 = m3.fit(arr3a, arr3b, epochs=4000)
history4 = m4.fit(arr4a, arr4b, epochs=4000)
history5 = m5.fit(arr5a, arr5b, epochs=4000)
# history6 = m6.fit(arr6a, arr6b, epochs=4000)


print(history1.history.keys())

plt.plot(history1.history['accuracy'], label="Model 1")
plt.plot(history2.history['accuracy'], label="Model 2")
plt.plot(history3.history['accuracy'], label="Model 3")
plt.plot(history4.history['accuracy'], label="Model 4")
plt.plot(history5.history['accuracy'], label="Model 5")
# plt.plot(history6.history['accuracy'], label="Model 6")

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
# plt.plot(history6.history['loss'], label="Model 6")

plt.title('Model Loss on 1000 data points, set 3')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left', borderaxespad=0.)
plt.show()


m1.save("m1c")
m2.save("m2c")
m3.save("m3c")
m4.save("m4c")
m5.save("m5c")
# m6.save("m6c")