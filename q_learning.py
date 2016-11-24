from __future__ import print_function
__author__ = "Abhimanyu Banerjee"

from random import random, choice
import matplotlib.pyplot as plt
import numpy as np
import timeit
from sys import exit
from copy import deepcopy
import pdb

from games import tttAgent2D
from games.board import empty_state, is_game_over

def play(player1, player2):
	state = empty_state()
	num_cells = len(state)*len(state[0])
	for turn in range(num_cells):
		if turn % 2 == 0:
			i, j = player1.action(deepcopy(state))
			symbol = player1.symbol
		else:
			i, j = player2.action(deepcopy(state))
			symbol = player2.symbol
		state[i][j] = symbol
		winner = int(is_game_over(state))
		if winner != 0: #one of the players won
			return winner 
	
	return winner #nobody won

def driver():
	player1 = tttAgent2D(symbol=1)
	player2 = tttAgent2D(symbol=-1)
	#print("\nNumber of possible states: ", player1.count_states())

	num_games = 1000
	for i in range(num_games):
		print("\nStarting game", i+1)
		winner = play(player1, player2)
		if winner == 0:
			print("Game ended in a draw!")
		else:
			if winner == 1:
				player_symbol = "X"
			else:
				player_symbol = "O"
			print(player_symbol, " won!")

driver()

