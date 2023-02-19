import os
import pickle
import numpy as np
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

def model2():
     model = Sequential()
     model.add(Dense(32, input_dim=7, activation='tanh'))
     model.add(Activation('tanh'))
     model.add(Dropout(0.05))
     model.add(Dense(150, activation='tanh'))
     # model.add(Activation('tanh'))
     model.add(Dropout(0.05))
     model.add(Dense(75, activation='tanh'))
     # model.add(Activation('tanh'))
     model.add(Dropout(0.05))
     model.add(Dense(50, activation='tanh'))
     # model.add(Activation('tanh'))
     model.add(Dropout(0.05))
     model.add(Dense(25, activation='tanh'))
     # model.add(Activation('tanh'))
     model.add(Dropout(0.05))
     model.add(Dense(6, activation='tanh'))
     # model.add(Activation('tanh'))
     model.add(Dropout(0.05))
     # model.add(Dense(6, init='uniform', bias=False, activation='tanh'))
     model.add(Dense(6))

     # model.add(Activation('tanh'))

     opt = Optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)

     model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

     return model

def model3():
     first = 7
     hidden = [500, 500]
     last = 6
     model = Sequential()
     # First layer
     layer = Dense(input_dim=first,
              units=hidden[0],
              kernel_initializer='random_normal',
              use_bias=True,
              bias_initializer='random_normal',
              activation='tanh')
     model.add(layer)

     # Hidden layers
     for i in range(len(hidden) - 1):
          layer = Dense(units=hidden[i + 1],
                   kernel_initializer='random_normal',
                   use_bias=True,
                   bias_initializer='random_normal',
                   activation='tanh')
          model.add(layer)

     # Last layer
     layer = Dense(units=last,
              kernel_initializer='random_normal',
              use_bias=True,
              bias_initializer='random_normal',
              activation='tanh')
     model.add(layer)

     model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

     return model

def model4():
    hidden = [500, 500]

    model = Sequential()
    layer = Dense(input_dim=first,
                  units=hidden[0],
                  kernel_initializer='random_normal',
                  use_bias=True,
                  bias_initializer='random_normal')
    model.add(layer)

    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(last))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    return model

def model5():

    model = Sequential()
    model.add(Dense(first))
    model.add(Dense(100, use_bias=True, activation='tanh'))
    model.add(Dense(100, use_bias=True, activation='tanh'))
    model.add(Dense(100, use_bias=True, activation='tanh'))
    model.add(Dense(last, use_bias=True, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    return model

def model6():
    model = Sequential()
    model.add(Dense(first))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.05))

    model.add(Dense(last))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    return model

m1 = model1()
m2 = model2()
m3 = model3()
m4 = model4()
m5 = model5()
m6 = model6()



scoreTotal1 = 0
for i in range(0, 1001):
    print("The iteration is: %d" % i)
    res1 = m1.predict(X[i:i+1, :7])
    sc1 = mean_absolute_error(Y[i:i+1, :7], res1)
    print(sc1)
    val = np.square(sc1)
    print(val)
    scoreTotal1 = scoreTotal1 + val
    print(scoreTotal1)
l = np.sum(scoreTotal1)
o = math.sqrt(l)
p = o/1000     # e_bar

f = open('p1.pckl', 'wb')
pickle.dump(p, f)
f.close()

scoreTotal1 = 0
for i in range(0, 1001):
    print("The iteration is: %d" % i)
    res1 = m2.predict(X[i:i+1, :7])
    sc1 = mean_absolute_error(Y[i:i+1, :7], res1)
    print(sc1)
    val = np.square(sc1)
    print(val)
    scoreTotal1 = scoreTotal1 + val
    print(scoreTotal1)
l = np.sum(scoreTotal1)
o = math.sqrt(l)
p = o/1000     # e_bar

f = open('p2.pckl', 'wb')
pickle.dump(p, f)
f.close()

scoreTotal1 = 0
for i in range(0, 1001):
    print("The iteration is: %d" % i)
    res1 = m3.predict(X[i:i+1, :7])
    sc1 = mean_absolute_error(Y[i:i+1, :7], res1)
    print(sc1)
    val = np.square(sc1)
    print(val)
    scoreTotal1 = scoreTotal1 + val
    print(scoreTotal1)
l = np.sum(scoreTotal1)
o = math.sqrt(l)
p = o/1000     # e_bar

f = open('p3.pckl', 'wb')
pickle.dump(p, f)
f.close()

scoreTotal1 = 0
for i in range(0, 1001):
    print("The iteration is: %d" % i)
    res1 = m4.predict(X[i:i+1, :7])
    sc1 = mean_absolute_error(Y[i:i+1, :7], res1)
    print(sc1)
    val = np.square(sc1)
    print(val)
    scoreTotal1 = scoreTotal1 + val
    print(scoreTotal1)
l = np.sum(scoreTotal1)
o = math.sqrt(l)
p = o/1000     # e_bar

f = open('p4.pckl', 'wb')
pickle.dump(p, f)
f.close()

scoreTotal1 = 0
for i in range(0, 1001):
    print("The iteration is: %d" % i)
    res1 = m5.predict(X[i:i+1, :7])
    sc1 = mean_absolute_error(Y[i:i+1, :7], res1)
    print(sc1)
    val = np.square(sc1)
    print(val)
    scoreTotal1 = scoreTotal1 + val
    print(scoreTotal1)
l = np.sum(scoreTotal1)
o = math.sqrt(l)
p = o/1000     # e_bar

f = open('p5.pckl', 'wb')
pickle.dump(p, f)
f.close()

scoreTotal1 = 0
for i in range(0, 1001):
    print("The iteration is: %d" % i)
    res1 = m6.predict(X[i:i+1, :7])
    sc1 = mean_absolute_error(Y[i:i+1, :7], res1)
    print(sc1)
    val = np.square(sc1)
    print(val)
    scoreTotal1 = scoreTotal1 + val
    print(scoreTotal1)
l = np.sum(scoreTotal1)
o = math.sqrt(l)
p = o/1000     # e_bar

f = open('p6.pckl', 'wb')
pickle.dump(p, f)
f.close()