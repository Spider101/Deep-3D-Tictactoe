from __future__ import print_function
__author__ = "Abhimanyu Banerjee"

from random import random, choice
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import timeit
from sys import exit
from copy import deepcopy
import pdb
import pickle

from games import tttAgent2D
from games.board import empty_state, is_game_over

def measure_performance(player1, player2, num_games):
	
	probs = [0,0,0]
	player1.behaviour_threshold = 1 #aggressive player
	player2.behaviour_threshold = 0 #random player
	for i in range(num_games):
		print("\nStarting game", i+1)
		winner = player1.self_play(player2)
		if winner == 0:
			print("Game ended in a draw!")
			probs[1] += 1.0/num_games
		else:
			if winner == 1:
				player_symbol = "X"
				probs[2] += 1.0/num_games
			else:
				player_symbol = "O"
				probs[0] += 1.0/num_games
			
			print(player_symbol, " won!")

	return probs

def driver():
	
	player1 = tttAgent2D(symbol=1, behaviour_threshold=0.05)
	player2 = tttAgent2D(symbol=-1, behaviour_threshold=0.05)
	batches, batch_probs = 5000, []

	for i in range(batches):
		print("\nStarting batch", i)
		if i % 10 == 0:
			batch_probs.append(measure_performance(deepcopy(player1), deepcopy(player2), 100))
		player1.self_play(player2)

	epochs = [i for i in range(int(batches/10))]
	win_probs = [probs[2] for probs in batch_probs]
	plt.plot(epochs, win_probs, label="Win Probability", color="g")
	plt.xlabel('Epochs')
	plt.ylabel('Probability')

	behaviour_threshold = player1.behaviour_threshold if player1.behaviour_threshold != 0 else player2.behaviour_threshold
	plt.title('RL Agent Performance vs. Random Agent\n({0} behaviour_threshold)'.format(behaviour_threshold))
	plt.legend()
	plt.savefig('selfplay_threshold_{0}.png'.format(behaviour_threshold))

	if player1.behaviour_threshold != 0.0:
		pickle.dump(player1.state_values, open("state_table_X.p", "wb"))

	if player2.behaviour_threshold != 0.0:
		pickle.dump(player2.state_values, open("state_table_O.p", "wb"))

driver()

