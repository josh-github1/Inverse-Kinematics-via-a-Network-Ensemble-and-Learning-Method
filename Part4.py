#EXTRACTING 2ND SET OF DATA FROM ORIGINAL TRAINING DATA

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


f1 = open('p1.pckl', 'rb')
p1 = pickle.load(f1)
f1.close()

f2 = open('p2.pckl', 'rb')
p2 = pickle.load(f2)
f2.close()

f3 = open('p3.pckl', 'rb')
p3 = pickle.load(f3)
f3.close()

f4 = open('p4.pckl', 'rb')
p4 = pickle.load(f4)
f4.close()

f5 = open('p5.pckl', 'rb')
p5 = pickle.load(f5)
f5.close()

f6 = open('p6.pckl', 'rb')
p6 = pickle.load(f6)
f6.close()



a = None
b = None
for i in range(0, 1000):
    print("The iteration is: %d" % i)
    res1 = m1.predict(X[i:i+1, :7])
    score1 = mean_absolute_error(Y[i:i+1, :6], res1)
    print("The score is: %f" % score1)
    if (score1 > p1 + scoref_1) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > p1 + scoref_1:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("2tr1X", "wb")
file2 = open("2tr1Y", "wb")
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
    if (score1 > p2 + scoref_2) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > p2 + scoref_2:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("2tr2X", "wb")
file2 = open("2tr2Y", "wb")
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
    if (score1 > p3 + scoref_3) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > p3 + scoref_3:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("2tr3X", "wb")
file2 = open("2tr3Y", "wb")
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
    if (score1 > p4 + scoref_4) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > p4 + scoref_4:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("2tr4X", "wb")
file2 = open("2tr4Y", "wb")
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
    if (score1 > p5 + scoref_5) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > p5 + scoref_5:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("2tr5X", "wb")
file2 = open("2tr5Y", "wb")
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
    if (score1 > p6 + scoref_6) and a is not None:
        print("This score is greater than the mean.")
        a = np.concatenate((a, X[i:i+1, :7]))
        b = np.concatenate((b, Y[i:i+1, :7]))
    else:
        print("It's not")
    if a is None and score1 > p6 + scoref_6:
        a = np.array(X[i:i+1, :7])
        b = np.array(Y[i:i+1, :7])
# open a binary file in write mode
file1 = open("2tr6X", "wb")
file2 = open("2tr6Y", "wb")
# save array to the file
np.save(file1, a)
np.save(file2, b)
# close the file
file1.close
file2.close
