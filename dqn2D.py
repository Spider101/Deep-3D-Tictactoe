from __future__ import print_function
__author__ = "Abhimanyu Banerjee"

import numpy as np
np.random.seed(1337)  # for reproducibility

from random import choice
from keras.models import Sequential
from keras.layers import Dense, Input
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K
import pdb

from memory import ExperienceReplay

from games import tttAgent2D
from games.board import empty_state, is_game_over, flatten_state, open_spots

class DQNAgent(object):

	def __init__(self, model, memory=None, memory_size=1000):
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)

		self.model = model

	''', epsilon_rate=0.5'''
	def train_network(self, game, num_epoch=1000, batch_size=50, gamma=0.9, epsilon=.1, reset_memory=False, observe=2):

		model = self.model
		nb_actions = model.output_shape[-1]
		win_count = 0
		
		for epoch in range(num_epoch):
			
			loss, winner = 0., None
			current_state = empty_state() #reset game
			game_over = False
			#self.clear_frames()
			
			if reset_memory:
				self.memory.reset_memory()
			
			while not game_over:

				#pdb.set_trace()
				if np.random.random() < epsilon: #or epoch < observe:
					empty_cells = open_spots(current_state)
					move = choice(empty_cells) # choose move randomly from available moves
				
				#choose the action for which Q function gives max value
				else: 
					q = model.predict(flatten_state(current_state))
					move = int(np.argmax(q[0]))

				next_state, reward = game.play_board(current_state, move)
				#reward = game.get_reward()
				
				#check who, if anyone, has won
				winner = is_game_over(next_state)
				print("Winner: ", winner)
				if winner != 0:
					game_over = True

				'''reward,'''
				transition = [flatten_state(current_state), move, reward, flatten_state(next_state), game_over] 
				self.memory.remember(*transition)
				current_state = next_state #update board state
				
				if epoch % observe == 0:
					
					batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets))
				
				'''if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == num_epoch):
					model.save_weights('weights.dat')'''
			
			if winner == -1*game.symbol: #ttt agent's symbol is inverted to get the model's symbol
				win_count += 1
			
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {}".format(epoch + 1, num_epoch, loss, epsilon, win_count))



def modelSetup(I, H, L):
	model = Sequential()
	model.add(Dense(H, input_dim=I, activation="relu"))
	
	for layer in range(L-1):
		model.add(Dense(H, activation="relu"))

	model.add(Dense(output_dim=I, activation="softmax"))
	model.compile(loss='mean_squared_error', optimizer='sgd')

	print(model.summary())

	return model

if __name__ == "__main__":

	# hyperparameters
	board_size = 3
	hidden_layer_neurons = 200 # number of hidden layer neurons
	hidden_layers = 1
	batch_size = 10 # every how many episodes to do a param update?
	learning_rate = 1e-4
	gamma = 0.99 # discount factor for reward
	decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

	input_size = board_size ** 2
	model = modelSetup(input_size, hidden_layer_neurons, hidden_layers)
	tictactoe = tttAgent2D(symbol=-1, is_learning=False)

	dqnAgent = DQNAgent(model=model, memory_size=-1)
	dqnAgent.train_network(tictactoe, batch_size=64, num_epoch=10, gamma=0.8)

