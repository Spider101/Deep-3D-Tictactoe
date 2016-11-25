from __future__ import print_function
__author__ = "Abhimanyu Banerjee"

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import random
from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K
import matplotlib.pyplot as plt

from games import tttAgent2D
from games.board import empty_state, is_game_over

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

model = Sequential()
model.add(Input(shape=something))
model.add(Dense(H, activation="relu"))
model.add(Dense(output_dim=something, activation="softmax"))
model.compile(loss='mean_squared_error', optimizer='sgd')

print(model.summary())

def policy_forward(model, input):
	
	activations = []
	model.predict(input) #forward pass

	for layer in model.layers:
		activations.append(layer.output)
	
	return activations[:-1], activations[-1]

def policy_backward(eph, epdlogp):
	
	""" backward pass. (eph is array of intermediate hidden states) """
	