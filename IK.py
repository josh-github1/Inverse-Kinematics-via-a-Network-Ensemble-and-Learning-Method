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
from keras.layers import Dropout\
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

# clf = KerasRegressor(build_fn=model2, epochs=500, batch_size=20, verbose=2)
# history = clf.fit(X, Y)
# res = clf.predict(X_test)


m1 = model1()
history1 = m1.fit(X, Y, epochs=1000)

m2 = model2()
history2 = m2.fit(X, Y, epochs=1000)

m3 = model3()
history3 = m3.fit(X, Y, epochs=1000)

m4 = model4()
history4 = m4.fit(X, Y, epochs=1000)

m5 = model5()
history5 = m5.fit(X, Y, epochs=1000)

m6 = model6()
history6 = m6.fit(X, Y, epochs=1000)

print(history1.history.keys())

plt.plot(history1.history['accuracy'], label="Model 1")
plt.plot(history2.history['accuracy'], label="Model 2")
plt.plot(history3.history['accuracy'], label="Model 3")
plt.plot(history4.history['accuracy'], label="Model 4")
plt.plot(history5.history['accuracy'], label="Model 5")
plt.plot(history6.history['accuracy'], label="Model 6")

plt.title('Model Accuracy on 1000 data points')
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

plt.title('Model Loss on 1000 data points')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left', borderaxespad=0.)
plt.show()

# model1 - 1.91 loss for 500 epochs
# model2 - 1.74 loss for 500 epochs            1.7 loss for 1000 epochs.
# model3 - 1.6592 loss for 500 epochs

m1.save("m1")
m2.save("m2")
m3.save("m3")
m4.save("m4")
m5.save("m5")
m6.save("m6")