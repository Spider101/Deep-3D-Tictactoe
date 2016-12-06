from __future__ import print_function
__author__ = "Abhimanyu Banerjee"

import numpy as np
np.random.seed(1337)  # for reproducibility

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy

from random import choice
from keras.models import Sequential
from keras.layers import Dense, Input
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K
import pdb

from memory import ExperienceReplay

from agents import TDAgent
from games.board import empty_state, is_game_over, flatten_state, open_spots

class DQNAgent(object):

	def __init__(self, model, memory=None, memory_size=1000):
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)

		self.model = model

	''', epsilon_rate=0.5'''
	def train_network(self, game, num_epochs=1000, batch_size=50, gamma=0.9, epsilon=[.1, 1], epsilon_rate=0.5, reset_memory=False, observe=2):

		model = self.model
		game.reset_agent()
		nb_actions = model.output_shape[-1]
		win_count, loss_count, total_reward = 0, 0, 0.0
		batch_probs, avg_reward = [], []
		delta = (epsilon[1] - epsilon[0]) /(num_epochs*epsilon_rate)
		epsilon_set = np.arange(epsilon[0], epsilon[1], delta)
		
		for epoch in range(1, num_epochs+1):
			
			loss, winner = 0., None
			current_state = empty_state(game.dims) #reset game
			game_over = False
			#self.clear_frames()
			
			if reset_memory:
				self.memory.reset_memory()
			
			if epoch % (num_epochs/1000) == 0:
				batch_probs.append(self.measure_performance(game, 100))
			
			while not game_over:

				if np.random.random() > epsilon_set[int(epoch*epsilon_rate - 0.5)]: #or epoch < observe:
					empty_cells = open_spots(current_state)
					move = choice(empty_cells) # choose move randomly from available moves
				
				#choose the action for which Q function gives max value
				else: 
					q = model.predict(flatten_state(current_state))
					move = int(np.argmax(q[0]))

				next_state, reward = game.play_board(deepcopy(current_state), move)
				
				#total_reward += reward
				#avg_reward.append(total_reward / epoch)

				#check who, if anyone, has won
				if reward != 0 or len(open_spots(next_state)) == 1:
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
				
				'''if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == num_epochs):
					model.save_weights('weights.dat')'''
			
			if reward == -1*game.symbol: #ttt agent's symbol is inverted to get the model's symbol
				win_count += 1
			
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon Rate {:.2f} | Wins {}".format(epoch, num_epochs, loss, epsilon_rate, win_count))

		epochs = [i for i in range(1000)]
		win_probs = [probs[2] for probs in batch_probs]
		
		plt.plot(epochs, win_probs, label="Win Probability", color="g")
		#plt.plot(epochs, avg_reward, label="Average Reward", color="r")
		plt.xlabel('Epochs')
		plt.ylabel('Probability')
		#plt.ylabel('Average Reward')

		plt.title('DQN Agent Performance v/s Q-Learning Agent (epsilon-rate={0})'.format(epsilon_rate))
		plt.legend()
		plt.savefig('dqn_epsilon-rate={0}.png'.format(epsilon_rate))
		plt.close()

		#ratio of number of games won [sum of win ratios * 100] to number of games played [100000 <- constant]
		return (np.array(win_probs).sum() / 1000) 


	def play_game(self, game):
			
		#set up game config
		model = self.model
		current_state = empty_state(game.dims) #reset game

		while True:
			#pdb.set_trace()
			q_values = model.predict(flatten_state(current_state))
			move = int(np.argmax(q_values[0]))

			next_state, reward = game.play_board(deepcopy(current_state), move)
			
			if np.array_equal(current_state, next_state):
				return -1

			'''while np.array_equal(current_state, next_state):
				q_values[:, move] = 0
				move = int(np.argmax(q_values[0]))
				next_state, reward = game.play_board(deepcopy(current_state), move)'''

			#check who, if anyone, has won
			if reward != 0 or len(open_spots(next_state)) == 1:
				return reward

			current_state = next_state

	def measure_performance(self, game, num_games):
		
		probs, games_played = [0,0,0], 0

		for i in range(num_games):
			
			winner = self.play_game(game)
			
			if winner != -1:
			
				games_played += 1	
				if winner == 0:
					probs[1] += 1.0
				elif winner == 1:
					probs[2] += 1.0
				else:
					probs[0] += 1.0
			
		if games_played > 0:
			probs[0] = probs[0] * 1. / games_played
			probs[1] = probs[1] * 1. / games_played
			probs[2] = probs[2] * 1. / games_played
		
		return probs 

def modelSetup(I, H, L):
	
	model = Sequential()
	model.add(Dense(H, input_dim=I, activation="relu"))
	
	for layer in range(L-1):
		model.add(Dense(H, activation="relu"))

	model.add(Dense(output_dim=I, activation="sigmoid"))
	model.compile(loss='mean_squared_error', optimizer='sgd')

	print(model.summary())

	return model

if __name__ == "__main__":

	# hyperparameters
	board_size = 3
	hidden_layer_neurons = 512 # number of hidden layer neurons
	hidden_layers = 1
	batch_size = 32 # every how many episodes to do a param update?
	learning_rate = 1e-4
	gamma = 0.99 # discount factor for reward
	decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

	input_size = board_size ** 2
	model = modelSetup(input_size, hidden_layer_neurons, hidden_layers)
	tdAgent2D = TDAgent(symbol=-1, is_learning=False, dims=2)

	win_ratios = []

	dqnAgent = DQNAgent(model=model, memory_size=5000)
	#win_ratios.append(dqnAgent.train_network(tictactoe, batch_size=batch_size, num_epochs=10000, gamma=gamma))
	#win_ratios.append(dqnAgent.train_network(tictactoe, batch_size=batch_size, num_epochs=10000, gamma=gamma, epsilon_rate=1))
	#win_ratios.append(dqnAgent.train_network(tictactoe, batch_size=batch_size, num_epochs=10000, gamma=gamma, epsilon_rate=0.8))
	win_ratios.append(dqnAgent.train_network(tdAgent2D, batch_size=batch_size, num_epochs=1000, gamma=gamma, epsilon_rate=0.3))

	print("Win Ratios: ", win_ratios)


