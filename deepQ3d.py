from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import random
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt







ticTacToeShape = (3,3,3)


def checkIfGameOver(state):
    """
    check if a game is over.
    This function will go to each element, and check if its an 'x' or an 'o'

    :param state: a valid state with strictly 3 dimensions, but can have any shape
    :return: true if game is over , false if game is not over
    """
    for i in xrange(ticTacToeShape[0]):
        for j in xrange(ticTacToeShape[1]):
            for k in xrange(ticTacToeShape[2]):
                if state[i][j][k] != 'x' and state[i][j][k] != 'o':
                    return False
    return True


def getOpenSpots(state):
    """
    gets the list of all open spots for a given state.
    Each element in the list will be a tuple , with ith,jth,kth positions
    :param state:
    :return:
    """
    openSpots = []

    for i in xrange(ticTacToeShape[0]):
        for j in xrange(ticTacToeShape[1]):
            for k in xrange(ticTacToeShape[2]):
                if state[i][j][k] != 'x' and state[i][j][k] != 'o':
                    openSpots.append((i,j,k))

    return openSpots

def makeRandomMove(state):
   """
   get a random tuple from the list of tuples
   :param state:
   :return:
   """
   return random.choice(getOpenSpots(state))






model = Sequential()
model.add(Dense(256,input_dim=ticTacToeShape[0]*ticTacToeShape[1]*ticTacToeShape[2]))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
