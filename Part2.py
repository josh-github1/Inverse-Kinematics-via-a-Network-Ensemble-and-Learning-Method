# EXTRACTING 1ST SET FROM ORIGINAL TRAINING DATA

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

res_1 = m1.predict(X)
scoreTr_mae = mean_absolute_error(Y, res_1)
scoreTr_rms = np.sqrt(mean_squared_error(Y, res_1))
scoref_1 = np.sqrt(mean_squared_error(Y, res_1))
# scoref_1 = scoreTr_rms/1000

res_2 = m2.predict(X)
score_2 = mean_absolute_error(Y, res_2)
scoreTr_rms2 = np.sqrt(mean_squared_error(Y, res_2))
scoref_2 = np.sqrt(mean_squared_error(Y, res_2))
# scoref_2 = scoreTr_rms2/1000

res_3 = m3.predict(X)
score_3 = mean_absolute_error(Y, res_3)
scoreTr_rms3 = np.sqrt(mean_squared_error(Y, res_3))
scoref_3 = np.sqrt(mean_squared_error(Y, res_3))
# scoref_3 = scoreTr_rms3/1000

res_4 = m4.predict(X)
score_4 = mean_absolute_error(Y, res_4)
scoreTr_rms4 = np.sqrt(mean_squared_error(Y, res_4))
scoref_4 = np.sqrt(mean_squared_error(Y, res_4))
# scoref_4 = scoreTr_rms4/1000

res_5 = m1.predict(X)
score_5 = mean_absolute_error(Y, res_5)
scoreTr_rms5 = np.sqrt(mean_squared_error(Y, res_5))
scoref_5 = np.sqrt(mean_squared_error(Y, res_5))
# scoref_5 = scoreTr_rms2/1000

res_6 = m1.predict(X)
score_6 = mean_absolute_error(Y, res_6)
scoreTr_rms6 = np.sqrt(mean_squared_error(Y, res_6))
scoref_6 = np.sqrt(mean_squared_error(Y, res_6))
# scoref_6 = scoreTr_rms6/1000

# Either score_fn or scoreTotaln would work the first time around since the errors are low
# that most

#scoreTotal1 = 0
#for i in range(0, 1001):
#    print("The iteration is: %d" % i)
#    res1 = m1.predict(X[i:i+1, :7])
#    sc1 = mean_absolute_error(Y[i:i+1, :7], res1)
#    print(sc1)
#    val = np.square(sc1)
#    print(val)
#    scoreTotal1 = scoreTotal1 + val
#    print(scoreTotal1)
#l = np.sum(scoreTotal1)
#o = math.sqrt(l)
#p = o/1000     # e_bar
#print(p)

a = None
b = None
for i in range(0, 1000):
    print("The iteration is: %d" % i)
    res1 = m1.predict(X[i:i+1, :7])
    score1 = mean_absolute_error(Y[i:i+1, :6], res1)
    print("The score is: %f" % score1)
    if (score1 > scoref_1) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > scoref_1:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("tr1X", "wb")
file2 = open("tr1Y", "wb")
# save array to the file
np.save(file1, a)
np.save(file2, b)
# close the file
file1.close
file2.close


a = None
b = None
for i in range(0, 1000):
    print("The iteration is: %d" % i)
    res1 = m2.predict(X[i:i+1, :7])
    score1 = mean_absolute_error(Y[i:i+1, :6], res1)
    print("The score is: %f" % score1)
    if (score1 > scoref_2) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > scoref_2:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("tr2X", "wb")
file2 = open("tr2Y", "wb")
# save array to the file
np.save(file1, a)
np.save(file2, b)
# close the file
file1.close
file2.close

a = None
b = None
for i in range(0, 1000):
    print("The iteration is: %d" % i)
    res1 = m3.predict(X[i:i+1, :7])
    score1 = mean_absolute_error(Y[i:i+1, :6], res1)
    print("The score is: %f" % score1)
    if (score1 > scoref_3) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > scoref_3:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("tr3X", "wb")
file2 = open("tr3Y", "wb")
# save array to the file
np.save(file1, a)
np.save(file2, b)
# close the file
file1.close
file2.close

a = None
b = None
for i in range(0, 1000):
    print("The iteration is: %d" % i)
    res1 = m4.predict(X[i:i+1, :7])
    score1 = mean_absolute_error(Y[i:i+1, :6], res1)
    print("The score is: %f" % score1)
    if (score1 > scoref_4) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > scoref_4:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("tr4X", "wb")
file2 = open("tr4Y", "wb")
# save array to the file
np.save(file1, a)
np.save(file2, b)
# close the file
file1.close
file2.close

a = None
b = None
for i in range(0, 1000):
    print("The iteration is: %d" % i)
    res1 = m5.predict(X[i:i+1, :7])
    score1 = mean_absolute_error(Y[i:i+1, :6], res1)
    print("The score is: %f" % score1)
    if (score1 > scoref_5) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > scoref_5:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("tr5X", "wb")
file2 = open("tr5Y", "wb")
# save array to the file
np.save(file1, a)
np.save(file2, b)
# close the file
file1.close
file2.close

a = None
b = None
for i in range(0, 1000):
    print("The iteration is: %d" % i)
    res1 = m6.predict(X[i:i+1, :7])
    score1 = mean_absolute_error(Y[i:i+1, :6], res1)
    print("The score is: %f" % score1)
    if (score1 > scoref_6) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > scoref_6:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("tr6X", "wb")
file2 = open("tr6Y", "wb")
# save array to the file
np.save(file1, a)
np.save(file2, b)
# close the file
file1.close
file2.close
