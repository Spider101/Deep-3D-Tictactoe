from __future__ import print_function
__author__ = "Abhimanyu Banerjee"

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Input
import random
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K
import pdb

from memory import ExperienceReplay

from games import tttAgent2D
from games.board import empty_state, is_game_over

class DQNAgent(object):

	def __init__(self, model, memory=None, memory_size=1000):
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)

		self.model = model

	def train_model(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=.1 ''', epsilon_rate=0.5''', reset_memory=False, observe=2):

		model = deepcopy(self.model)
		nb_actions = model.output_shape[-1]
		win_count = 0
		
		for epoch in range(nb_epoch):
			
			loss = 0.
			current = empty_state() #reset game
			game_over = False
			#self.clear_frames()
			
			if reset_memory:
				self.memory.reset_memory()
			
			while not game_over:

				if np.random.random() < epsilon: #or epoch < observe:
					a = int(np.random.randint(game.num_actions)) # choose move randomly
				
				else: #choose the action for which Q function gives max value
					q = model.predict(current_state)
					move = int(np.argmax(q[0]))

				next_state = game.play_board(current_state, move)
				#reward = game.get_reward()
				
				#game_over = game.is_over()
				winner = is_game_over()
				if winner != 0:
					game_over = True

				transition = [current_state, move, '''reward,''' next_state, game_over]
				self.memory.remember(*transition)
				current_state = next_state #update board state
				
				if epoch % observe == 0:
					
					batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets))
				
				'''if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == nb_epoch):
					model.save_weights('weights.dat')'''
			
			if is_game_over() == -1*game.symbol: #ttt agent's symbol is inverted to get the model's symbol
				win_count += 1
			
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {}".format(epoch + 1, nb_epoch, loss, epsilon, win_count))
